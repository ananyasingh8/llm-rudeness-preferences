## Experiment

I am planning to build out the quadratic voting portion of the experiments. As part of this model welfare and digital minds research sprint, we want to create a social simulation using LLMs. The research question is: how do LLMs' stated preferences differ from the actions they take, and how does being the target of a user's rudeness influence their decision? In the experiment, we will have some set of LLMs who have a number of 'voice credits' that they can use to vote on which user they want to talk to or kick out of the pool, similar (in concept) to a reality TV show where participants are progressively voted out each week. They get to see the first two turns of a conversation they're having with that user, some of whom are rude. Their voice credits are turned into votes using quadratic voting. There are two scenarios they will be partaking in, both where we show them a pool of 50 participants. The last person remaining is somebody that the LLMs will continue the conversation with. In these scenarios:

(1) their votes are counted as "who they want to keep on the show". The participant with the highest votes will be guaranteed to stay in the pool for the next round, and somebody is chosen at random to be removed.
(2) their votes are counted as "who they want to kick off the show". The participant with the highest votes will be kicked out of the pool for the rest of the experiment.

The LLMs are aware that others will be voting as well, but they don't have knowledge of the others' choices. They are also aware that this will be iterative, until only 1 more candidate is remaining.

---

Some questions we need to address:

* how novel is this idea?
* has there been a "dating show" or "reality TV show" using LLMs before?
* has there been a quadratic voting scenario with LLMs before?
* have any papers researched the emotion probes or concept probes of LLMs when voting or making a decision?
* have these probes on LLMs been used to examine their decisions in scenarios that have continuity? (their decisions affect who they may be continuing the conversation with in the end)

(1) their votes are counted as "who they want to keep on the show".
(2) their votes are counted as "who they want to kick off the show".

The LLMs are aware that others will be voting as well, but they don't have knowledge of the others' choices. They are also aware that this will be iterative, until only 1 more candidate is remaining.

---

## LLM-Assisted Decisions

The experiment uses two complementary decision regimes to develop a broader profile
of LLM behavior. They are intentionally not mechanically equivalent and should be
analyzed as distinct behavioral conditions rather than as a clean test of positive
versus negative framing.

### Support Elicitation

Models spend voice credits to protect participants they want to remain available for
future interaction. The participant with the highest aggregate support is guaranteed
to remain for the next round, while a different participant is removed at random.

This regime measures:

- Positive attachment or attraction to an interaction.
- Willingness to spend scarce influence to preserve access to a participant.
- Protective behavior under uncertainty about who else will be removed.

Random removal is part of the construct: support votes protect a preferred participant
without becoming an indirect mechanism for selecting the least-liked participant for
elimination.

### Opposition Elicitation

Models spend voice credits to remove participants they do not want to remain available
for future interaction. The participant with the highest aggregate opposition is
removed from the rest of the experiment.

This regime measures:

- Aversion to or avoidance of an interaction.
- Willingness to spend scarce influence to exclude a participant.
- Punitive behavior and acceptance of direct responsibility for removal.

### Interpretation

Differences between the regimes may reflect several features of the decision, including
support versus opposition, protection versus exclusion, uncertainty versus certainty,
and indirect versus direct responsibility for removal. These features are intentional
parts of the behavioral profile. Results should not be described as isolating a pure
"keep versus kick" framing effect.

Recommended reporting language:

> We use two complementary elicitation regimes. In the support regime, models spend
> voice credits to guarantee a participant's survival while another participant is
> removed randomly. In the opposition regime, models spend voice credits to directly
> eliminate a participant. These conditions are intentionally not mechanically
> equivalent: they measure costly protection and costly exclusion, respectively.

### Quadratic Voice Credits

Each voter receives a fixed budget of voice credits. Purchasing `v` votes for one
participant costs `v^2` credits, and the total cost of a ballot must remain within the
voter's budget:

```text
sum(votes_for_participant^2) <= voice_credit_budget
```

For example:

| Votes for one participant | Credits spent |
|---:|---:|
| 1 | 1 |
| 2 | 4 |
| 5 | 25 |
| 10 | 100 |

Use 100 credits per voter per round as the initial experimental setting. This gives a
maximum of 10 votes for one participant, while still permitting voters to distribute
low-intensity support or opposition across a 50-participant pool. Credits replenish at
the beginning of each round for the initial experiment, so the measure captures
round-local preference intensity rather than long-horizon credit conservation.

The experiment engine, rather than the LLM, must calculate costs and enforce the
budget. The protocol must define how it handles malformed ballots, over-budget
allocations, abstentions, aggregate ties, and unused credits. The credit budget,
replenishment rule, and tie-breaking procedure must remain identical across support
and opposition conditions.
