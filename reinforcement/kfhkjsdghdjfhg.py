class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        return self.weights * self.featExtractor.getFeatures(state,action)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # getValue from nextState returns max_a' q in the next state
        # getQValue returns the sum of feature_i times weight_i for 0 < i < n
        difference = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
        featureVector = self.featExtractor.getFeatures(state, action)

        # for state, action in featureVector.keys():
        for feature in featureVector.keys():
            self.weights[feature] = self.weights[feature] + self.alpha * difference * featureVector[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # nope, it's fine
            pass
