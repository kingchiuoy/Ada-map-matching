/*
 *  Licensed to GraphHopper GmbH under one or more contributor
 *  license agreements. See the NOTICE file distributed with this work for
 *  additional information regarding copyright ownership.
 *
 *  GraphHopper GmbH licenses this file to you under the Apache License,
 *  Version 2.0 (the "License"); you may not use this file except in
 *  compliance with the License. You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package com.graphhopper.matching;

import com.bmw.hmm.SequenceState;
import com.bmw.hmm.Transition;
import com.bmw.hmm.ViterbiAlgorithm;
import com.graphhopper.GraphHopper;
import com.graphhopper.config.LMProfile;
import com.graphhopper.config.Profile;
import com.graphhopper.routing.*;
import com.graphhopper.routing.lm.LMApproximator;
import com.graphhopper.routing.lm.LandmarkStorage;
import com.graphhopper.routing.lm.PrepareLandmarks;
import com.graphhopper.routing.querygraph.QueryGraph;
import com.graphhopper.routing.querygraph.VirtualEdgeIteratorState;
import com.graphhopper.routing.util.DefaultEdgeFilter;
import com.graphhopper.routing.util.TraversalMode;
import com.graphhopper.routing.weighting.Weighting;
import com.graphhopper.storage.Graph;
import com.graphhopper.storage.index.LocationIndexTree;
import com.graphhopper.storage.index.Snap;
import com.graphhopper.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import java.util.*;
import java.util.stream.Collectors;

/**
 * This class matches real world GPX entries to the digital road network stored
 * in GraphHopper. The Viterbi algorithm is used to compute the most likely
 * sequence of map matching candidates. The Viterbi algorithm takes into account
 * the distance between GPX entries and map matching candidates as well as the
 * routing distances between consecutive map matching candidates.
 * <p>
 * <p>
 * See http://en.wikipedia.org/wiki/Map_matching and Newson, Paul, and John
 * Krumm. "Hidden Markov map matching through noise and sparseness." Proceedings
 * of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic
 * Information Systems. ACM, 2009.
 *
 * @author Peter Karich
 * @author Michael Zilske
 * @author Stefan Holder
 * @author kodonnell
 */
public class MapMatching {

    private final Logger logger = LoggerFactory.getLogger(getClass());

    private final Graph graph;
    private final PrepareLandmarks landmarks;
    private final LocationIndexTree locationIndex;
    private double measurementErrorSigma = 50.0;
    private double transitionProbabilityBeta = 2.0;
    private final int maxVisitedNodes;
    private final DistanceCalc distanceCalc = new DistancePlaneProjection();
    private final Weighting weighting;
    private QueryGraph queryGraph;

    public MapMatching(GraphHopper graphHopper, PMap hints) {
        this.locationIndex = (LocationIndexTree) graphHopper.getLocationIndex();

        if (hints.has("vehicle"))
            throw new IllegalArgumentException("MapMatching hints may no longer contain a vehicle, use the profile parameter instead, see core/#1958");
        if (hints.has("weighting"))
            throw new IllegalArgumentException("MapMatching hints may no longer contain a weighting, use the profile parameter instead, see core/#1958");

        if (graphHopper.getProfiles().isEmpty()) {
            throw new IllegalArgumentException("No profiles found, you need to configure at least one profile to use map matching");
        }
        if (!hints.has("profile")) {
            throw new IllegalArgumentException("You need to specify a profile to perform map matching");
        }
        String profileStr = hints.getString("profile", "");
        Profile profile = graphHopper.getProfile(profileStr);
        if (profile == null) {
            List<Profile> profiles = graphHopper.getProfiles();
            List<String> profileNames = new ArrayList<>(profiles.size());
            for (Profile p : profiles) {
                profileNames.add(p.getName());
            }
            throw new IllegalArgumentException("Could not find profile '" + profileStr + "', choose one of: " + profileNames);
        }

        boolean disableLM = hints.getBool(Parameters.Landmark.DISABLE, false);
        if (graphHopper.getLMPreparationHandler().isEnabled() && disableLM && !graphHopper.getRouterConfig().isLMDisablingAllowed())
            throw new IllegalArgumentException("Disabling LM is not allowed");

        boolean disableCH = hints.getBool(Parameters.CH.DISABLE, false);
        if (graphHopper.getCHPreparationHandler().isEnabled() && disableCH && !graphHopper.getRouterConfig().isCHDisablingAllowed())
            throw new IllegalArgumentException("Disabling CH is not allowed");

        // see map-matching/#177: both ch.disable and lm.disable can be used to force Dijkstra which is the better
        // (=faster) choice when the observations are close to each other
        boolean useDijkstra = disableLM || disableCH;

        if (graphHopper.getLMPreparationHandler().isEnabled() && !useDijkstra) {
            // using LM because u-turn prevention does not work properly with (node-based) CH
            List<String> lmProfileNames = new ArrayList<>();
            PrepareLandmarks lmPreparation = null;
            for (LMProfile lmProfile : graphHopper.getLMPreparationHandler().getLMProfiles()) {
                lmProfileNames.add(lmProfile.getProfile());
                if (lmProfile.getProfile().equals(profile.getName())) {
                    lmPreparation = graphHopper.getLMPreparationHandler().getPreparation(
                            lmProfile.usesOtherPreparation() ? lmProfile.getPreparationProfile() : lmProfile.getProfile()
                    );
                }
            }
            if (lmPreparation == null) {
                throw new IllegalArgumentException("Cannot find LM preparation for the requested profile: '" + profile.getName() + "'" +
                        "\nYou can try disabling LM using " + Parameters.Landmark.DISABLE + "=true" +
                        "\navailable LM profiles: " + lmProfileNames);
            }
            landmarks = lmPreparation;
        } else {
            landmarks = null;
        }
        graph = graphHopper.getGraphHopperStorage();
        weighting = graphHopper.createWeighting(profile, hints);

//        Reduce the search radius for quicker calculation.
        int myMaxNodes = 3000;

//        this.maxVisitedNodes = hints.getInt(Parameters.Routing.MAX_VISITED_NODES, Integer.MAX_VALUE);
        this.maxVisitedNodes = hints.getInt(Parameters.Routing.MAX_VISITED_NODES, myMaxNodes);
    }

    /**
     * Beta parameter of the exponential distribution for modeling transition
     * probabilities.
     */
    public void setTransitionProbabilityBeta(double transitionProbabilityBeta) {
        this.transitionProbabilityBeta = transitionProbabilityBeta;
    }

    /**
     * Standard deviation of the normal distribution [m] used for modeling the
     * GPS error.
     */
    public void setMeasurementErrorSigma(double measurementErrorSigma) {
        this.measurementErrorSigma = measurementErrorSigma;
    }

    public MatchResult match(List<Observation> observations) {
        List<Observation> filteredObservations = filterObservations(observations);

        // Snap observations to links. Generates multiple candidate snaps per observation.
        // In the next step, we will turn them into splits, but we already call them splits now
        // because they are modified in place.
        List<Collection<Snap>> splitsPerObservation = filteredObservations.stream().map(o -> locationIndex.findNClosest(o.getPoint().lat, o.getPoint().lon, DefaultEdgeFilter.allEdges(weighting.getFlagEncoder()), measurementErrorSigma))
                .collect(Collectors.toList());

        // Create the query graph, containing split edges so that all the places where an observation might have happened
        // are a node. This modifies the Snap objects and puts the new node numbers into them.
        queryGraph = QueryGraph.create(graph, splitsPerObservation.stream().flatMap(Collection::stream).collect(Collectors.toList()));

        // Due to how LocationIndex/QueryGraph is implemented, we can get duplicates when a point is snapped
        // directly to a tower node instead of creating a split / virtual node. No problem, but we still filter
        // out the duplicates for performance reasons.
        splitsPerObservation = splitsPerObservation.stream().map(this::deduplicate).collect(Collectors.toList());

        // Creates candidates from the Snaps of all observations (a candidate is basically a
        // Snap + direction).
        List<ObservationWithCandidateStates> timeSteps = createTimeSteps(filteredObservations, splitsPerObservation);

        // Compute the most likely sequence of map matching candidates:
        List<SequenceState<State, Observation, Path>> seq = computeViterbiSequence(timeSteps);

        List<EdgeIteratorState> path = seq.stream().filter(s1 -> s1.transitionDescriptor != null).flatMap(s1 -> s1.transitionDescriptor.calcEdges().stream()).collect(Collectors.toList());

        MatchResult result = new MatchResult(prepareEdgeMatches(seq));
        result.setMergedPath(new MapMatchedPath(queryGraph, weighting, path));
        result.setMatchMillis(seq.stream().filter(s -> s.transitionDescriptor != null).mapToLong(s -> s.transitionDescriptor.getTime()).sum());
        result.setMatchLength(seq.stream().filter(s -> s.transitionDescriptor != null).mapToDouble(s -> s.transitionDescriptor.getDistance()).sum());
        result.setGPXEntriesLength(gpxLength(observations));
        result.setGraph(queryGraph);
        result.setWeighting(weighting);
        return result;
    }

    /**
     * Filters observations to only those which will be used for map matching (i.e. those which
     * are separated by at least 2 * measurementErrorSigman
     */
    private List<Observation> filterObservations(List<Observation> observations) {
        List<Observation> filtered = new ArrayList<>();
        Observation prevEntry = null;
        int last = observations.size() - 1;
        for (int i = 0; i <= last; i++) {
            Observation observation = observations.get(i);
            if (i == 0 || i == last || distanceCalc.calcDist(
                    prevEntry.getPoint().getLat(), prevEntry.getPoint().getLon(),
                    observation.getPoint().getLat(), observation.getPoint().getLon()) > 2 * measurementErrorSigma) {
                filtered.add(observation);
                prevEntry = observation;
            } else {
                logger.debug("Filter out observation: {}", i + 1);
            }
        }
        return filtered;
    }

    private Collection<Snap> deduplicate(Collection<Snap> splits) {
        // Only keep one split per node number. Let's say the last one.
        Map<Integer, Snap> splitsByNodeNumber = splits.stream().collect(Collectors.toMap(Snap::getClosestNode, s -> s, (s1, s2) -> s2));
        return splitsByNodeNumber.values();
    }

    /**
     * Creates TimeSteps with candidates for the GPX entries but does not create emission or
     * transition probabilities. Creates directed candidates for virtual nodes and undirected
     * candidates for real nodes.
     */
    private List<ObservationWithCandidateStates> createTimeSteps(List<Observation> filteredObservations, List<Collection<Snap>> splitsPerObservation) {
        if (splitsPerObservation.size() != filteredObservations.size()) {
            throw new IllegalArgumentException(
                    "filteredGPXEntries and queriesPerEntry must have same size.");
        }

        final List<ObservationWithCandidateStates> timeSteps = new ArrayList<>();
        for (int i = 0; i < filteredObservations.size(); i++) {
            Observation observation = filteredObservations.get(i);
            Collection<Snap> splits = splitsPerObservation.get(i);
            List<State> candidates = new ArrayList<>();
            for (Snap split : splits) {
                if (queryGraph.isVirtualNode(split.getClosestNode())) {
                    List<VirtualEdgeIteratorState> virtualEdges = new ArrayList<>();
                    EdgeIterator iter = queryGraph.createEdgeExplorer().setBaseNode(split.getClosestNode());
                    while (iter.next()) {
                        if (!queryGraph.isVirtualEdge(iter.getEdge())) {
                            throw new RuntimeException("Virtual nodes must only have virtual edges "
                                    + "to adjacent nodes.");
                        }
                        virtualEdges.add((VirtualEdgeIteratorState) queryGraph.getEdgeIteratorState(iter.getEdge(), iter.getAdjNode()));
                    }
                    if (virtualEdges.size() != 2) {
                        throw new RuntimeException("Each virtual node must have exactly 2 "
                                + "virtual edges (reverse virtual edges are not returned by the "
                                + "EdgeIterator");
                    }

                    // Create a directed candidate for each of the two possible directions through
                    // the virtual node. We need to add candidates for both directions because
                    // we don't know yet which is the correct one. This will be figured
                    // out by the Viterbi algorithm.
                    candidates.add(new State(observation, split, virtualEdges.get(0), virtualEdges.get(1)));
                    candidates.add(new State(observation, split, virtualEdges.get(1), virtualEdges.get(0)));
                } else {
                    // Create an undirected candidate for the real node.
                    candidates.add(new State(observation, split));
                }
            }

            timeSteps.add(new ObservationWithCandidateStates(observation, candidates));
        }
        return timeSteps;
    }

    /**
     * Computes the most likely state sequence for the observations.
     */
    private List<SequenceState<State, Observation, Path>> computeViterbiSequence(List<ObservationWithCandidateStates> timeSteps) {
        final HmmProbabilities probabilities = new HmmProbabilities(measurementErrorSigma, transitionProbabilityBeta);
        final ViterbiAlgorithm<State, Observation, Path> viterbi = new ViterbiAlgorithm<>();

        int timeStepCounter = 0;
        ObservationWithCandidateStates prevTimeStep = null;
        for (ObservationWithCandidateStates timeStep : timeSteps) {
            final Map<State, Double> emissionLogProbabilities = new HashMap<>();
            Map<Transition<State>, Double> transitionLogProbabilities = new HashMap<>();
            Map<Transition<State>, Path> roadPaths = new HashMap<>();
            for (State candidate : timeStep.candidates) {
                // distance from observation to road in meters
                final double distance = candidate.getSnap().getQueryDistance();
                emissionLogProbabilities.put(candidate, probabilities.emissionLogProbability(distance));
            }

            if (prevTimeStep == null) {
                viterbi.startWithInitialObservation(timeStep.observation, timeStep.candidates, emissionLogProbabilities);
            } else {
                final double linearDistance = distanceCalc.calcDist(prevTimeStep.observation.getPoint().lat,
                        prevTimeStep.observation.getPoint().lon, timeStep.observation.getPoint().lat, timeStep.observation.getPoint().lon);

                for (State from : prevTimeStep.candidates) {
                    for (State to : timeStep.candidates) {
                        final Path path = createRouter().calcPath(from.getSnap().getClosestNode(), to.getSnap().getClosestNode(), from.isOnDirectedEdge() ? from.getOutgoingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE, to.isOnDirectedEdge() ? to.getIncomingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE);
                        if (path.isFound()) {
                            double transitionLogProbability = probabilities.transitionLogProbability(path.getDistance(), linearDistance);
                            Transition<State> transition = new Transition<>(from, to);
                            roadPaths.put(transition, path);
                            transitionLogProbabilities.put(transition, transitionLogProbability);
                        }
                    }
                }
                viterbi.nextStep(timeStep.observation, timeStep.candidates,
                        emissionLogProbabilities, transitionLogProbabilities,
                        roadPaths);
            }
            if (viterbi.isBroken()) {
                fail(timeStepCounter, prevTimeStep, timeStep);
            }

            timeStepCounter++;
            prevTimeStep = timeStep;
        }

        return viterbi.computeMostLikelySequence();
    }


    public MatchResult matchWithSpecificMethod(List<Observation> observations, int trackNum){
        boolean isNear = false;
        int startIndex = 0, endIndex = observations.size() - 1;

        for(Observation o : observations) {
            Collection<Snap> tmp = locationIndex.
                    findNClosest(o.getPoint().lat, o.getPoint().lon, DefaultEdgeFilter.allEdges(weighting.getFlagEncoder()),
                            measurementErrorSigma);

            for(Snap snap : tmp) {
                if(snap.getQueryDistance() < 10) {
                    isNear = true;
                    break;
                }
            }
            if(isNear) break;
            startIndex++;
        }
        isNear = false;
        for(; endIndex >= 0; endIndex--) {
            Observation o = observations.get(endIndex);
            Collection<Snap> tmp = locationIndex.
                    findNClosest(o.getPoint().lat, o.getPoint().lon, DefaultEdgeFilter.allEdges(weighting.getFlagEncoder()),
                            measurementErrorSigma);

            for(Snap snap : tmp) {
                if(snap.getQueryDistance() < 10) {
                    isNear = true;
                    break;
                }
            }
            if(isNear) break;
        }

        observations = observations.subList(startIndex, endIndex + 1);

        List<Observation> filteredObservations = filterObservations(observations);

//        System.out.println(observationPoints);
        // Snap observations to links. Generates multiple candidate snaps per observation.
        // In the next step, we will turn them into splits, but we already call them splits now
        // because they are modified in place.
        List<Collection<Snap>> splitsPerObservation;
        splitsPerObservation = filteredObservations.stream().map(o -> locationIndex.
                findNClosest(o.getPoint().lat, o.getPoint().lon, DefaultEdgeFilter.allEdges(weighting.getFlagEncoder()),
                        measurementErrorSigma))
                .collect(Collectors.toList());

        // Create the query graph, containing split edges so that all the places where an observation might have happened
        // are a node. This modifies the Snap objects and puts the new node numbers into them.
        queryGraph = QueryGraph.create(graph, splitsPerObservation.stream().flatMap(Collection::stream).collect(Collectors.toList()));

        // Due to how LocationIndex/QueryGraph is implemented, we can get duplicates when a point is snapped
        // directly to a tower node instead of creating a split / virtual node. No problem, but we still filter
        // out the duplicates for performance reasons.
        splitsPerObservation = splitsPerObservation.stream().map(this::deduplicate).collect(Collectors.toList());

        // Creates candidates from the Snaps of all observations (a candidate is basically a
        // Snap + direction).

        List<ObservationWithCandidateStates> timeSteps = createTimeSteps(filteredObservations, splitsPerObservation);

        // Compute the most likely sequence of map matching candidates:


        List<MatchResult> results = new ArrayList<>();
        List<SequenceState<State, Observation, Path>> seq;
        StopWatch computeSW = new StopWatch().start();
        seq = adaMatch(timeSteps);
        computeSW.stop();

        List<EdgeIteratorState> path = seq.stream().filter(s1 -> s1.transitionDescriptor != null).flatMap(s1 -> s1.transitionDescriptor.calcEdges().stream()).collect(Collectors.toList());

        MatchResult result = new MatchResult(prepareEdgeMatches(seq));
        result.setMergedPath(new MapMatchedPath(queryGraph, weighting, path));
        result.setMatchMillis(seq.stream().filter(s -> s.transitionDescriptor != null).mapToLong(s -> s.transitionDescriptor.getTime()).sum());
        result.setMatchLength(seq.stream().filter(s -> s.transitionDescriptor != null).mapToDouble(s -> s.transitionDescriptor.getDistance()).sum());
        result.setGPXEntriesLength(gpxLength(observations));
        result.setGraph(queryGraph);
        result.setWeighting(weighting);
        results.add(result);

        return result;
    }


    private List<SequenceState<State, Observation, Path>> adaMatch(List<ObservationWithCandidateStates> timeSteps) {
//        System.out.println("Using Confidence Abb MEthod");
        final HmmProbabilities probabilities = new HmmProbabilities(measurementErrorSigma, transitionProbabilityBeta);
        ObservationWithCandidateStates prevTimeStep = null;
        StopWatch loopWatch = new StopWatch();
        StopWatch buildWatch = new StopWatch();
        StopWatch backWardTime = new StopWatch();
        StopWatch calTime = new StopWatch();
        StopWatch preTime = new StopWatch();

        StopWatch speedTime = new StopWatch();
        StopWatch pathTime = new StopWatch();
        StopWatch dirTime = new StopWatch();
        StopWatch posTime = new StopWatch();
        StopWatch afterTime = new StopWatch();
        List<Integer> candidateList = new ArrayList<>();
        int stepCounter = 0;
//        System.out.println("TimeStepSize: " + timeSteps.size());
        int countReduce = 0;
        int inValidSnappedCount = 0;
        loopWatch.start();

        int windowSize = 5;
        DescriptiveStatistics speedStatics = new DescriptiveStatistics(windowSize);

//        maxVisitedNodes = 6000;



        for (ObservationWithCandidateStates timeStep : timeSteps) {
            final Map<State, Double> emissionLogProbabilities = new HashMap<>();
//            Map<Transition<State>, Double> transitionLogProbabilities = new HashMap<>();
            if(timeStep.observation.accuracy > 50) {
//                System.out.println(stepCounter +  ": Worng Observation");
                stepCounter++;
                continue;
            }
            //            The window size of the average speed


            int metricNum = 3;
            int candidateCount = 0;
            if (prevTimeStep == null) {
                timeStep.prev = null;
                for (State candidate : timeStep.candidates) {
                    candidate.setProb(1.0 / (double) timeStep.candidates.size());
                }
                speedStatics.addValue(timeStep.observation.getSpeed());

            } else {
                candidateList.add(prevTimeStep.candidates.size());
                timeStep.prev = prevTimeStep;
//              This is the distance of the Priori Probability
                final double linearDistance = distanceCalc.calcDist(prevTimeStep.observation.getPoint().lat,
                        prevTimeStep.observation.getPoint().lon, timeStep.observation.getPoint().lat, timeStep.observation.getPoint().lon);
                double observationDirection = prevTimeStep.observation.getDirection();
                double timeDis = Math.ceil((double)(timeStep.observation.getTimestep() - prevTimeStep.observation.getTimestep()));
                preTime.start();
                speedStatics.addValue(timeStep.observation.getSpeed());
                double avgSpeed = speedStatics.getMean();
                if(Math.abs(avgSpeed) < 5.0) {
                    avgSpeed = 5.0;
                }
                timeStep.avgSpeed = avgSpeed;
                timeStep.timeDis = timeDis;
                List<State> toDelete = new ArrayList<>();
                double connectionToDeleteBound = 0.005;
                List<Double> validProb = new ArrayList<>();

                //                Speed Probability

//                The calculation of the Speed Probability.
                int curCount = 0;

                for(State to : timeStep.candidates) {
                    speedTime.start();
                    double prob = 0.0;
                    double speedProb;
                    int prevCount = 0;
                    State maxPrev = null;
                    double maxProb = Double.NEGATIVE_INFINITY;



                    for (State from : prevTimeStep.candidates) {
                        double dist = Double.MAX_VALUE;
                        pathTime.start();
                        final Path path = createRouter().calcPath(from.getSnap().getClosestNode(), to.getSnap().getClosestNode(), from.isOnDirectedEdge() ? from.getOutgoingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE, to.isOnDirectedEdge() ? to.getIncomingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE);
                        pathTime.stop();
                        if (path.isFound()) {
                            speedProb = Utils.getAdjustSpeedProb(path.getDistance() / (double)timeDis, avgSpeed);
                            dist = path.getDistance();
                        } else {
                            speedProb = 0.0;
                        }

                        double checkProb = 1.0 / dist * speedProb;
                        if(checkProb > maxProb) {
//                            System.out.println("test");
                        }
                        if(checkProb * from.getProb() > maxProb) {
                            maxPrev = from;
                            maxProb = checkProb * from.getProb();
                        }
                        prob += speedProb * from.getProb();
                        prevCount++;
                    }

                    if(maxPrev == null) {
//                        System.out.println("Check Here");
                    }

                    to.maxPrev = maxPrev;
                    to.speedScore = prob;
                    to.setProb(prob);
                    if(prob < connectionToDeleteBound) {
                        toDelete.add(to);
                        speedTime.stop();
                    } else {
                        validProb.add(prob);
                        speedTime.stop();

                        dirTime.start();
//                        Direction
                        PointList pointList = null;

                        double pathDirection = 0.0;
                        double preLat,postLat, postLon, preLon;

                        if(to.isOnDirectedEdge()) {
                            pointList = to.getIncomingVirtualEdge().fetchWayGeometry(FetchMode.ALL);
                            if(to.getSnap().getWayIndex() >= pointList.size()) {
//                            System.out.println("For bug");
                            }
                        } else {
                            EdgeIteratorState tmp = to.getSnap().getClosestEdge();

                            pointList = tmp.fetchWayGeometry(FetchMode.ALL);
                        }

                        if (pointList.size() >= 2) {
                            int ind = to.isOnDirectedEdge() ? 0 :  to.getSnap().getWayIndex();
                            if(ind == 0) {
                                preLat = pointList.getLat(ind);
                                preLon = pointList.getLon(ind);
                                postLat = pointList.getLat(ind + 1);
                                postLon = pointList.getLon(ind + 1);
                            } else {
                                preLat = pointList.getLat(ind - 1);
                                preLon = pointList.getLon(ind - 1);
                                postLat = pointList.getLat(ind);
                                postLon = pointList.getLon(ind);
                            }
                            pathDirection = Utils.calDirection(preLat, preLon, postLat, postLon);
                            if(to.isOnDirectedEdge()) {
                                boolean test = to.getIncomingVirtualEdge().getReverse(EdgeIteratorState.REVERSE_STATE);
                                if(!test) {
                                    pathDirection = Utils.calDirection(postLat, postLon, preLat, preLon);
                                }
                            } else {}
                        } else {
                            inValidSnappedCount++;
                            pathDirection = Double.MAX_VALUE;
                        }
                        double dirProb = Distributions.standardNormalDistribution(Math.abs(pathDirection - observationDirection) / 120.0) * Math.sqrt(2.0);
                        to.dirScore = dirProb;
                        dirTime.stop();

//                        Position Prob
                        // distance from observation to road in meters
                        posTime.start();
                        final double distance = to.getSnap().getQueryDistance();
                        double disProb;
                        final double standardDistance = Math.abs(distance) / measurementErrorSigma;
                        disProb = Distributions.standardNormalDistribution(standardDistance) * Math.sqrt(2.0);
                        to.posScore = disProb;
                        posTime.stop();
                    }
                    candidateCount++;
                    curCount++;
                }
                afterTime.start();
//                System.out.println(finalProbs.stream().mapToDouble(a->a).summaryStatistics());
                for(State s : toDelete) {
                    timeStep.candidates.remove(s);
                }

                if(timeStep.candidates.size() == 0) {
                    stepCounter++;
                    continue;
                }

                double[][] probMatrix = new double[metricNum][timeStep.candidates.size()];
                candidateCount = 0;
                for(State to : timeStep.candidates) {
                    probMatrix[1][candidateCount] = to.speedScore;
                    probMatrix[2][candidateCount] = to.dirScore;
                    probMatrix[0][candidateCount] = to.posScore;
                    candidateCount++;
                }
                afterTime.stop();
                preTime.stop();
                candidateCount = 0;

                calTime.start();
                RealVector confidence = Compute.getResult(probMatrix, timeStep.candidates.size(), metricNum);
                double[] confindenceArr = confidence.toArray();
                RealVector weightVector =  confidence.getSubVector(0, 3);
                double score = 0.0;
                RealMatrix S = MatrixUtils.createRealMatrix(probMatrix);
                for(int i = 0; i < S.getColumnDimension(); i++) {
                    score += weightVector.dotProduct(S.getColumnVector(i)) * confindenceArr[metricNum + i];
                }

                candidateCount = 0;

                for (State to : timeStep.candidates) {
                    to.setProb(confindenceArr[metricNum + candidateCount]);
                    candidateCount++;
                }

                calTime.stop();

                boolean isDeleteFinished = false;
                double sumProbs = 1.0;
                while (!isDeleteFinished) {
                    isDeleteFinished = true;
                    ArrayList<State> delList = new ArrayList<>();
                    for (State to : timeStep.candidates) {
                        if (to.getProb() / sumProbs < 0.05) {
                            delList.add(to);
                            isDeleteFinished = false;
                        }
                        to.setProb(to.getProb() / sumProbs);
                    }
                    if (!isDeleteFinished) {
                        for (State t : delList) {
                            sumProbs -= t.getProb();
                            timeStep.candidates.remove(t);
                        }
                    }
                }

                if (score < 0.45)  {
                    stepCounter++;
                    timeStep.candidates.clear();
                    countReduce++;
                    continue;
                }
            }
            stepCounter++;
            prevTimeStep = timeStep;
        }

        stepCounter = 0;

        buildWatch.start();
        List<SequenceState<State, Observation, Path>> result = new ArrayList<>();
        State prev = null;
        int ind = timeSteps.size() - 1;
        while(prev == null || prev.maxPrev == null) {
            if(ind >= 0 && ind < timeSteps.size() &&  timeSteps.get(ind).candidates.size() != 0) {
                prev = getMaxState(timeSteps.get(ind).candidates, 1);
            } else if(ind < 0) {
                break;
            }
            ind--;
        }

        while(prev != null && prev.maxPrev != null) {
            StringBuilder tmp = new StringBuilder();
            tmp.append(prev.getSnap().getSnappedPoint().lat);
            tmp.append(",");
            tmp.append(prev.getSnap().getSnappedPoint().lon);
            final Path path = createRouter().calcPath(prev.maxPrev.getSnap().getClosestNode(), prev.getSnap().getClosestNode(), prev.maxPrev.isOnDirectedEdge() ? prev.maxPrev.getOutgoingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE,
                    prev.isOnDirectedEdge() ? prev.getIncomingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE);
            result.add(new SequenceState<State, Observation, Path>(prev.maxPrev, prev.maxPrev.getEntry(),
                    path));
            prev = prev.maxPrev;
        }
        Collections.reverse(result);
        return result;
    }


    private List<SequenceState<State, Observation, Path>> adaMatchWithBacktrack(List<ObservationWithCandidateStates> timeSteps) {
//        System.out.println("Using Confidence Abb MEthod");
        final HmmProbabilities probabilities = new HmmProbabilities(measurementErrorSigma, transitionProbabilityBeta);
        ObservationWithCandidateStates prevTimeStep = null;
        StopWatch loopWatch = new StopWatch();
        StopWatch buildWatch = new StopWatch();
        StopWatch backWardTime = new StopWatch();
        StopWatch calTime = new StopWatch();
        StopWatch preTime = new StopWatch();

        StopWatch speedTime = new StopWatch();
        StopWatch pathTime = new StopWatch();
        StopWatch dirTime = new StopWatch();
        StopWatch posTime = new StopWatch();
        StopWatch afterTime = new StopWatch();
        List<Integer> candidateList = new ArrayList<>();
        int stepCounter = 0;
//        System.out.println("TimeStepSize: " + timeSteps.size());
        int countReduce = 0;
        int inValidSnappedCount = 0;
        loopWatch.start();

        int windowSize = 5;
        DescriptiveStatistics speedStatics = new DescriptiveStatistics(windowSize);

//        maxVisitedNodes = 6000;



        for (ObservationWithCandidateStates timeStep : timeSteps) {
            final Map<State, Double> emissionLogProbabilities = new HashMap<>();
//            Map<Transition<State>, Double> transitionLogProbabilities = new HashMap<>();
            if(timeStep.observation.accuracy > 50) {
//                System.out.println(stepCounter +  ": Worng Observation");
                stepCounter++;
                continue;
            }
            //            The window size of the average speed


            int metricNum = 3;
            int candidateCount = 0;
            if (prevTimeStep == null) {
                timeStep.prev = null;
                for (State candidate : timeStep.candidates) {
                    candidate.setProb(1.0 / (double) timeStep.candidates.size());
                }
                speedStatics.addValue(timeStep.observation.getSpeed());

            } else {
                candidateList.add(prevTimeStep.candidates.size());
                timeStep.prev = prevTimeStep;
//              This is the distance of the Priori Probability
                final double linearDistance = distanceCalc.calcDist(prevTimeStep.observation.getPoint().lat,
                        prevTimeStep.observation.getPoint().lon, timeStep.observation.getPoint().lat, timeStep.observation.getPoint().lon);
                double observationDirection = prevTimeStep.observation.getDirection();
                double timeDis = Math.ceil((double)(timeStep.observation.getTimestep() - prevTimeStep.observation.getTimestep()));
                preTime.start();
                speedStatics.addValue(timeStep.observation.getSpeed());
                double avgSpeed = speedStatics.getMean();
                if(Math.abs(avgSpeed) < 5.0) {
                    avgSpeed = 5.0;
                }
                timeStep.avgSpeed = avgSpeed;
                timeStep.timeDis = timeDis;
                List<State> toDelete = new ArrayList<>();
                double connectionToDeleteBound = 0.005;
                List<Double> validProb = new ArrayList<>();

                //                Speed Probability

//                The calculation of the Speed Probability.
                int curCount = 0;

                for(State to : timeStep.candidates) {
                    speedTime.start();
                    double prob = 0.0;
                    double speedProb;
                    int prevCount = 0;
                    State maxPrev = null;
                    double maxProb = Double.NEGATIVE_INFINITY;



                    for (State from : prevTimeStep.candidates) {
                        double dist = Double.MAX_VALUE;
                        pathTime.start();
                        final Path path = createRouter().calcPath(from.getSnap().getClosestNode(), to.getSnap().getClosestNode(), from.isOnDirectedEdge() ? from.getOutgoingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE, to.isOnDirectedEdge() ? to.getIncomingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE);
                        pathTime.stop();
                        if (path.isFound()) {
                            speedProb = Utils.getAdjustSpeedProb(path.getDistance() / (double)timeDis, avgSpeed);
                            dist = path.getDistance();
                        } else {
                            speedProb = 0.0;
                        }

                        double checkProb = 1.0 / dist * speedProb;
                        if(checkProb > maxProb) {
//                            System.out.println("test");
                        }
                        if(checkProb * from.getProb() > maxProb) {
                            maxPrev = from;
                            maxProb = checkProb * from.getProb();
                        }
                        prob += speedProb * from.getProb();
                        prevCount++;
                    }

                    if(maxPrev == null) {
//                        System.out.println("Check Here");
                    }

                    to.maxPrev = maxPrev;
                    to.speedScore = prob;
                    to.setProb(prob);
                    if(prob < connectionToDeleteBound) {
                        toDelete.add(to);
                        speedTime.stop();
                    } else {
                        validProb.add(prob);
                        speedTime.stop();

                        dirTime.start();
//                        Direction
                        PointList pointList = null;

                        double pathDirection = 0.0;
                        double preLat,postLat, postLon, preLon;

                        if(to.isOnDirectedEdge()) {
                            pointList = to.getIncomingVirtualEdge().fetchWayGeometry(FetchMode.ALL);
                            if(to.getSnap().getWayIndex() >= pointList.size()) {
//                            System.out.println("For bug");
                            }
                        } else {
                            EdgeIteratorState tmp = to.getSnap().getClosestEdge();

                            pointList = tmp.fetchWayGeometry(FetchMode.ALL);
                        }

                        if (pointList.size() >= 2) {
                            int ind = to.isOnDirectedEdge() ? 0 :  to.getSnap().getWayIndex();
                            if(ind == 0) {
                                preLat = pointList.getLat(ind);
                                preLon = pointList.getLon(ind);
                                postLat = pointList.getLat(ind + 1);
                                postLon = pointList.getLon(ind + 1);
                            } else {
                                preLat = pointList.getLat(ind - 1);
                                preLon = pointList.getLon(ind - 1);
                                postLat = pointList.getLat(ind);
                                postLon = pointList.getLon(ind);
                            }
                            pathDirection = Utils.calDirection(preLat, preLon, postLat, postLon);
                            if(to.isOnDirectedEdge()) {
                                boolean test = to.getIncomingVirtualEdge().getReverse(EdgeIteratorState.REVERSE_STATE);
                                if(!test) {
                                    pathDirection = Utils.calDirection(postLat, postLon, preLat, preLon);
                                }
                            } else {}
                        } else {
                            inValidSnappedCount++;
                            pathDirection = Double.MAX_VALUE;
                        }
                        double dirProb = Distributions.standardNormalDistribution(Math.abs(pathDirection - observationDirection) / 120.0) * Math.sqrt(2.0);
                        to.dirScore = dirProb;
                        dirTime.stop();

//                        Position Prob
                        // distance from observation to road in meters
                        posTime.start();
                        final double distance = to.getSnap().getQueryDistance();
                        double disProb;
                        final double standardDistance = Math.abs(distance) / measurementErrorSigma;
                        disProb = Distributions.standardNormalDistribution(standardDistance) * Math.sqrt(2.0);
                        to.posScore = disProb;
                        posTime.stop();
                    }
                    candidateCount++;
                    curCount++;
                }
                afterTime.start();
//                System.out.println(finalProbs.stream().mapToDouble(a->a).summaryStatistics());
                for(State s : toDelete) {
                    timeStep.candidates.remove(s);
                }

                if(timeStep.candidates.size() == 0) {
                    stepCounter++;
                    continue;
                }

                double[][] probMatrix = new double[metricNum][timeStep.candidates.size()];
                candidateCount = 0;
                for(State to : timeStep.candidates) {
                    probMatrix[1][candidateCount] = to.speedScore;
                    probMatrix[2][candidateCount] = to.dirScore;
                    probMatrix[0][candidateCount] = to.posScore;
                    candidateCount++;
                }
                afterTime.stop();
                preTime.stop();
                candidateCount = 0;

                calTime.start();
                RealVector confidence = Compute.getResult(probMatrix, timeStep.candidates.size(), metricNum);
                double[] confindenceArr = confidence.toArray();
                RealVector weightVector =  confidence.getSubVector(0, 3);
                double score = 0.0;
                RealMatrix S = MatrixUtils.createRealMatrix(probMatrix);
                for(int i = 0; i < S.getColumnDimension(); i++) {
                    score += weightVector.dotProduct(S.getColumnVector(i)) * confindenceArr[metricNum + i];
                }

                candidateCount = 0;

                for (State to : timeStep.candidates) {
                    to.setProb(confindenceArr[metricNum + candidateCount]);
                    candidateCount++;
                }

                calTime.stop();

                boolean isDeleteFinished = false;
                double sumProbs = 1.0;
                while (!isDeleteFinished) {
                    isDeleteFinished = true;
                    ArrayList<State> delList = new ArrayList<>();
                    for (State to : timeStep.candidates) {
                        if (to.getProb() / sumProbs < 0.05) {
                            delList.add(to);
                            isDeleteFinished = false;
                        }
                        to.setProb(to.getProb() / sumProbs);
                    }
                    if (!isDeleteFinished) {
                        for (State t : delList) {
                            sumProbs -= t.getProb();
                            timeStep.candidates.remove(t);
                        }
                    }
                }

                if (score < 0.45)  {
                    stepCounter++;
                    timeStep.candidates.clear();
                    countReduce++;
                    continue;
                }

                //                Backwards
                boolean isBackWardFinished = false;
                ObservationWithCandidateStates prev = prevTimeStep;
                ObservationWithCandidateStates cur = timeStep;
                Map<State, Double> stateToProb = new HashMap<>();
                while (!isBackWardFinished) {
                    isBackWardFinished = true;
                    final double otherlinearDistance = distanceCalc.calcDist(prev.observation.getPoint().lat,
                            prev.observation.getPoint().lon, cur.observation.getPoint().lat, cur.observation.getPoint().lon);
                    for (State from : prev.candidates) {
                        double prob = 0.0;
                        for (State to : cur.candidates) {
                            final Path path = createRouter().calcPath(from.getSnap().getClosestNode(),
                                    to.getSnap().getClosestNode(),
                                    from.isOnDirectedEdge() ? from.getOutgoingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE, to.isOnDirectedEdge() ? to.getIncomingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE);
                            if (path.isFound()) {
                                double tmp = Utils.getAdjustSpeedProb(path.getDistance() / (double)cur.timeDis, cur.avgSpeed);
                                prob += tmp * to.getProb();
                            }
                        }
                        stateToProb.put(from, prob);
                    }
                    double sumBackWardProbs = stateToProb.values().stream().reduce(0.0, (a, b) -> (a + b));

                    isDeleteFinished = false;
                    while (!isDeleteFinished) {
                        isDeleteFinished = true;
                        ArrayList<State> delList = new ArrayList<>();
                        for (State from : prev.candidates) {
                            double tmp = stateToProb.get(from) / sumBackWardProbs;
                            if (tmp < 0.05) {
                                delList.add(from);
                                isDeleteFinished = false;
                                isBackWardFinished = false;
                            }
                            from.setProb(tmp);
                        }
                        if (!isDeleteFinished) {
                            for (State t : delList) {
                                sumBackWardProbs -= stateToProb.get(t);
                                prev.candidates.remove(t);
                            }
                        }
                    }
                    stateToProb.clear();

//                    Change the maxPrev
                    for(State to : cur.candidates) {
                        double speedProb;
                        State maxPrev = null;
                        double maxProb = Double.NEGATIVE_INFINITY;
                        for (State from : prev.candidates) {
                            double dist = Double.MAX_VALUE;
                            pathTime.start();
                            final Path path = createRouter().calcPath(from.getSnap().getClosestNode(), to.getSnap().getClosestNode(), from.isOnDirectedEdge() ? from.getOutgoingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE, to.isOnDirectedEdge() ? to.getIncomingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE);
                            pathTime.stop();
                            if (path.isFound()) {
                                speedProb =  Utils.getAdjustSpeedProb(path.getDistance() / (double)cur.timeDis, cur.avgSpeed);;
                                dist = path.getDistance();
                            } else {
                                speedProb = 0.0;
                            }
                            double checkProb = 1.0 / dist * speedProb;
                            if (speedProb * from.getProb() > maxProb) {
                                maxPrev = from;
                                maxProb = speedProb * from.getProb();
                            }
                        }
                        to.maxPrev = maxPrev;
                    }
                    cur = cur.prev;
                    prev = cur.prev;
                    if (prev == null) break;
                }
            }
            stepCounter++;
            prevTimeStep = timeStep;
        }

        stepCounter = 0;

        buildWatch.start();
        List<SequenceState<State, Observation, Path>> result = new ArrayList<>();
        State prev = null;
        int ind = timeSteps.size() - 1;
        while(prev == null || prev.maxPrev == null) {
            if(ind >= 0 && ind < timeSteps.size() &&  timeSteps.get(ind).candidates.size() != 0) {
                prev = getMaxState(timeSteps.get(ind).candidates, 1);
            } else if(ind < 0) {
                break;
            }
            ind--;
        }

        while(prev != null && prev.maxPrev != null) {
            StringBuilder tmp = new StringBuilder();
            tmp.append(prev.getSnap().getSnappedPoint().lat);
            tmp.append(",");
            tmp.append(prev.getSnap().getSnappedPoint().lon);
            final Path path = createRouter().calcPath(prev.maxPrev.getSnap().getClosestNode(), prev.getSnap().getClosestNode(), prev.maxPrev.isOnDirectedEdge() ? prev.maxPrev.getOutgoingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE,
                    prev.isOnDirectedEdge() ? prev.getIncomingVirtualEdge().getEdge() : EdgeIterator.ANY_EDGE);
            result.add(new SequenceState<State, Observation, Path>(prev.maxPrev, prev.maxPrev.getEntry(),
                    path));
            prev = prev.maxPrev;
        }
        Collections.reverse(result);
        return result;
    }

    private State getMaxState(Collection<State> states, int count) {
//        System.out.println(count + " " + states.size());
        return states.stream().max(new Comparator<State>() {
            @Override
            public int compare(State o1, State o2) {
                double prob1 = o1.getProb(), prob2 = o2.getProb();
                if (prob1 == prob2) {
                    return 0;
                } else if (prob1 > prob2) {
                    return 1;
                } else {
                    return -1;
                }
            }
        }).get();
    }


    private void fail(int timeStepCounter, ObservationWithCandidateStates prevTimeStep, ObservationWithCandidateStates timeStep) {
        String likelyReasonStr = "";
        if (prevTimeStep != null) {
            double dist = distanceCalc.calcDist(prevTimeStep.observation.getPoint().lat, prevTimeStep.observation.getPoint().lon, timeStep.observation.getPoint().lat, timeStep.observation.getPoint().lon);
            if (dist > 2000) {
                likelyReasonStr = "Too long distance to previous measurement? "
                        + Math.round(dist) + "m, ";
            }
        }

        throw new IllegalArgumentException("Sequence is broken for submitted track at time step "
                + timeStepCounter + ". "
                + likelyReasonStr + "observation:" + timeStep.observation + ", "
                + timeStep.candidates.size() + " candidates: "
                + getSnappedCandidates(timeStep.candidates)
                + ". If a match is expected consider increasing max_visited_nodes.");
    }

    private BidirRoutingAlgorithm createRouter() {
        BidirRoutingAlgorithm router;
        if (landmarks != null) {
            AStarBidirection algo = new AStarBidirection(queryGraph, weighting, TraversalMode.EDGE_BASED) {
                @Override
                protected void initCollections(int size) {
                    super.initCollections(50);
                }
            };
            LandmarkStorage lms = landmarks.getLandmarkStorage();
            int activeLM = Math.min(8, lms.getLandmarkCount());
            algo.setApproximation(LMApproximator.forLandmarks(queryGraph, lms, activeLM));
            algo.setMaxVisitedNodes(maxVisitedNodes);
            router = algo;
        } else {
            router = new DijkstraBidirectionRef(queryGraph, weighting, TraversalMode.EDGE_BASED) {
                @Override
                protected void initCollections(int size) {
                    super.initCollections(50);
                }
            };
            router.setMaxVisitedNodes(maxVisitedNodes);
        }
        return router;
    }

    private List<EdgeMatch> prepareEdgeMatches(List<SequenceState<State, Observation, Path>> seq) {
        // This creates a list of directed edges (EdgeIteratorState instances turned the right way),
        // each associated with 0 or more of the observations.
        // These directed edges are edges of the real street graph, where nodes are intersections.
        // So in _this_ representation, the path that you get when you just look at the edges goes from
        // an intersection to an intersection.

        // Implementation note: We have to look at both states _and_ transitions, since we can have e.g. just one state,
        // or two states with a transition that is an empty path (observations snapped to the same node in the query graph),
        // but these states still happen on an edge, and for this representation, we want to have that edge.
        // (Whereas in the ResponsePath representation, we would just see an empty path.)

        // Note that the result can be empty, even when the input is not. Observations can be on nodes as well as on
        // edges, and when all observations are on the same node, we get no edge at all.
        // But apart from that corner case, all observations that go in here are also in the result.

        // (Consider totally forbidding candidate states to be snapped to a point, and make them all be on directed
        // edges, then that corner case goes away.)
        List<EdgeMatch> edgeMatches = new ArrayList<>();
        List<State> states = new ArrayList<>();
        EdgeIteratorState currentDirectedRealEdge = null;
        for (SequenceState<State, Observation, Path> transitionAndState : seq) {
            // transition (except before the first state)
            if (transitionAndState.transitionDescriptor != null) {
                for (EdgeIteratorState edge : transitionAndState.transitionDescriptor.calcEdges()) {
                    EdgeIteratorState newDirectedRealEdge = resolveToRealEdge(edge);
                    if (currentDirectedRealEdge != null) {
                        if (!equalEdges(currentDirectedRealEdge, newDirectedRealEdge)) {
                            EdgeMatch edgeMatch = new EdgeMatch(currentDirectedRealEdge, states);
                            edgeMatches.add(edgeMatch);
                            states = new ArrayList<>();
                        }
                    }
                    currentDirectedRealEdge = newDirectedRealEdge;
                }
            }
            // state
            if (transitionAndState.state.isOnDirectedEdge()) { // as opposed to on a node
                EdgeIteratorState newDirectedRealEdge = resolveToRealEdge(transitionAndState.state.getOutgoingVirtualEdge());
                if (currentDirectedRealEdge != null) {
                    if (!equalEdges(currentDirectedRealEdge, newDirectedRealEdge)) {
                        EdgeMatch edgeMatch = new EdgeMatch(currentDirectedRealEdge, states);
                        edgeMatches.add(edgeMatch);
                        states = new ArrayList<>();
                    }
                }
                currentDirectedRealEdge = newDirectedRealEdge;
            }
            states.add(transitionAndState.state);
        }
        if (currentDirectedRealEdge != null) {
            EdgeMatch edgeMatch = new EdgeMatch(currentDirectedRealEdge, states);
            edgeMatches.add(edgeMatch);
        }
        return edgeMatches;
    }

    private double gpxLength(List<Observation> gpxList) {
        if (gpxList.isEmpty()) {
            return 0;
        } else {
            double gpxLength = 0;
            Observation prevEntry = gpxList.get(0);
            for (int i = 1; i < gpxList.size(); i++) {
                Observation entry = gpxList.get(i);
                gpxLength += distanceCalc.calcDist(prevEntry.getPoint().lat, prevEntry.getPoint().lon, entry.getPoint().lat, entry.getPoint().lon);
                prevEntry = entry;
            }
            return gpxLength;
        }
    }

    private boolean equalEdges(EdgeIteratorState edge1, EdgeIteratorState edge2) {
        return edge1.getEdge() == edge2.getEdge()
                && edge1.getBaseNode() == edge2.getBaseNode()
                && edge1.getAdjNode() == edge2.getAdjNode();
    }

    private EdgeIteratorState resolveToRealEdge(EdgeIteratorState edgeIteratorState) {
        if (queryGraph.isVirtualNode(edgeIteratorState.getBaseNode()) || queryGraph.isVirtualNode(edgeIteratorState.getAdjNode())) {
            return graph.getEdgeIteratorStateForKey(((VirtualEdgeIteratorState) edgeIteratorState).getOriginalEdgeKey());
        } else {
            return edgeIteratorState;
        }
    }

    private String getSnappedCandidates(Collection<State> candidates) {
        String str = "";
        for (State gpxe : candidates) {
            if (!str.isEmpty()) {
                str += ", ";
            }
            str += "distance: " + gpxe.getSnap().getQueryDistance() + " to "
                    + gpxe.getSnap().getSnappedPoint();
        }
        return "[" + str + "]";
    }

    private static class MapMatchedPath extends Path {
        MapMatchedPath(Graph graph, Weighting weighting, List<EdgeIteratorState> edges) {
            super(graph);
            int prevEdge = EdgeIterator.NO_EDGE;
            for (EdgeIteratorState edge : edges) {
                addDistance(edge.getDistance());
                addTime(GHUtility.calcMillisWithTurnMillis(weighting, edge, false, prevEdge));
                addEdge(edge.getEdge());
                prevEdge = edge.getEdge();
            }
            if (edges.isEmpty()) {
                setFound(false);
            } else {
                setFromNode(edges.get(0).getBaseNode());
                setFound(true);
            }
        }
    }

}