## Experiment

I am planning to build out the quadratic voting portion of the experiments. As part of this model welfare and digital minds research sprint, we want to create a social simulation using LLMs. The research question is: how do LLMs' stated preferences differ from the actions they take, and how does being the target of a user's rudeness influence their decision? In the experiment, we will have some set of LLMs who have a number of 'voice credits' that they can use to vote on which user they want to talk to or kick out of the pool, similar (in concept) to a reality TV show where participants are progressively voted out each week. They get to see the first two turns of a conversation they're having with that user, some of whom are rude. Their voice credits are turned into votes using quadratic voting. There are two scenarios they will be partaking in, both where we show them a pool of 50 participants. In these scenarios:

(1) their votes are counted as "who they want to keep on the show".
(2) their votes are counted as "who they want to kick off the show".

The LLMs are aware that others will be voting as well, but they don't have knowledge of the others' choices. They are also aware that this will be iterative, until only 1 more candidate is remaining.
