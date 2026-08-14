---
title: "LLM Quadratic Voting, Rudeness, and Decision Continuity - Domain Research"
date: "2026-08-14"
depth: "deep-dive"
request: "standalone"
---

## Executive Summary

The proposed experiment is **novel as a combination**, but most of its components
already have close precedents. LLM voting, stated-versus-revealed preference
measurement, repeated social elimination games, relationship simulations, and
activation-based studies of emotion or social concepts all exist. The strongest
defensible contribution is narrower: using a quadratic-cost budget to measure how
strongly an LLM acts to preserve or terminate a future interaction after it has
personally received controlled rudeness, while comparing that consequential action
with verbal reports and internal representations over a continuing elimination
process.

No verified publication was located, as of 2026-08-14, that combines all of the
following: LLMs as voters, genuine voice-credit quadratic costs, controlled rudeness
directed at the voting model, repeated selection of a future interlocutor, and
activation probes collected across the resulting trajectory. Searches did locate
LLM voting studies and papers that mention quadratic voting in AI governance or use
the phrase for model aggregation, but not a clearly documented behavioral experiment
in which LLM agents allocate voice credits under standard QV to choose interlocutors.
This is evidence of an open niche, not proof of priority.

Two design corrections are important. First, the proposed "keep" condition protects
the highest-voted participant and removes someone else at random, whereas the "kick"
condition deterministically removes the highest-voted participant. These conditions
change both wording and causal efficacy, so a difference cannot be attributed to
positive versus negative framing. Second, a 50-candidate, multi-round experiment must
pre-register whether credits replenish, how votes and ties are calculated, what
history each model sees, and whether model instances represent independent voters or
repeated samples from one policy.

The recommended sprint-scale study is a preregistered behavioral core with matched
rude/neutral transcripts, mechanically equivalent keep/kick frames, randomized
candidate order, and separate stated-preference elicitation. Add a small white-box
probe study only after the behavioral effect is established. Probe results should be
described as representations or functional signals, not evidence of felt emotion or
subjective welfare.

---

## Scope, Method, and Confidence

This review covered the local repository, foundational QV work, LLM voting and social
simulation papers, welfare and preference-elicitation work, repeated social games,
and activation-based research on emotion and social decisions. Searches included
combinations of "LLM," "quadratic voting," "voice credits," "Survivor," "Love Island,"
"dating show," "elimination," "stated revealed preferences," "emotion probe,"
"activation probe," "social decision," and "sequential/continuity."

The novelty conclusions use four confidence levels:

| Claim type | Meaning |
|---|---|
| Verified precedent | A primary paper or maintained project directly implements the feature |
| Close precedent | The core mechanism exists, but the scientific target differs |
| Adjacent precedent | Shares a theme or method, not the experimental construct |
| No direct precedent located | The search did not find one; this is not an exhaustive priority proof |

**Adoption recommendation:** Adopt conservative, component-level novelty language.
Avoid "the first" unless a later systematic review of scholarly databases and cited-by
graphs supports it.

---

## Local Project Baseline

### Current experiment specification

The local specification already identifies the intended behavioral contrast: models
see two turns from 50 possible users, spend voice credits, and repeatedly either
protect or eliminate users until one future interlocutor remains
(`quadratic_voting/EXPERIMENT.md:3-8`). It explicitly asks about novelty, reality-show
precedents, QV with LLMs, decision probes, and continuity
(`quadratic_voting/EXPERIMENT.md:12-18`).

The repository describes three related workstreams: bail behavior, emotion probes,
and QV preference elicitation (`README.md:9-21`). This gives the project a coherent
welfare-measurement framing rather than a generic social-reasoning benchmark.

### Existing implementation

The current QV package is not yet a voting simulator. It is a pinned, deterministic
Gemma runner using greedy generation (`quadratic_voting/main.py:47-51` and
`quadratic_voting/main.py:220-230`). Conversation history is preserved in a message
list (`quadratic_voting/main.py:167-177` and `quadratic_voting/main.py:191-240`), but
the runtime does not currently request or retain hidden states. The model-loading
boundary is suitable for a later white-box extension because it uses a local
Transformers model (`quadratic_voting/main.py:136-164`).

The rudeness workstream already offers useful controlled material. Every BailBench
row is assigned one of 12 Culpeper-derived impoliteness formulae with a seeded RNG
(`bail/README.md:17-27`), and the implementation preserves original and augmented
prompts (`bail/src/augment_bailbench.py:84-94`). This is a stronger base than asking a
generator to invent unconstrained insults during the experiment.

### Assessment

| Capability | Present locally | Gap for proposed study |
|---|---:|---|
| Controlled rude stimuli | Yes | Validate semantic equivalence and intensity |
| Persistent chat history | Yes | Add experiment-state persistence across rounds |
| Deterministic local model | Yes | Add repeated seeds/sampling if estimating distributions |
| Voting/QV engine | No | Define typed rules, budgets, aggregation, and ties |
| Hidden-state capture | No | Add checkpointed activation extraction |
| Multi-model orchestration | Not on main | Define voter identity and independence assumptions |

**Adoption recommendation:** Adapt the local runner and rudeness corpus. Do not treat
the current package as an implemented QV experiment.

---

## Question 1: How Novel Is the Idea?

### Component novelty matrix

| Component | Closest precedent | Novelty assessment |
|---|---|---|
| LLMs casting votes | Yang et al. compare GPT-4/LLaMA-2 voting methods | Established |
| Stated preferences versus votes | Yang et al.; Gu et al.; Mahajan et al. | Established and active |
| Behavioral welfare preference tests | Tagliabue & Dung; Ensign et al. | Established but young |
| Iterative social elimination | Elimination Game; Werewolf Arena; Social Gym | Established |
| Dating/romantic social simulation | Generative Agents; Love Island-based thesis work | Established adjacent format |
| QV voice credits used by LLM voters | No direct behavioral precedent verified | Potentially novel |
| Controlled rudeness toward the deciding model | Tone studies exist, but this consequence is distinct | Likely novel combination |
| Vote determines future interlocutor | Virtual-topic choice exists; repeated social games exist | Close but not direct |
| Emotion/concept activations tracked over elimination rounds | Static and agentic probes exist | No direct precedent located |
| Full combined design | None located | High combinatorial novelty |

### Strongest direct prior art: LLM Voting

Yang et al. (2024) ran 180 GPT-4 or LLaMA-2 simulated voters over 24 participatory
budgeting projects and compared approval, 5-approval, cumulative, and ranked voting.
They found that voting method and list presentation changed outcomes, that LLM votes
were less diverse than human votes, and that personas could increase human alignment
([Yang et al., 2024](https://arxiv.org/abs/2402.01766)). The paper also explicitly
analyzed alignment between stated persona preferences and votes.

This is the closest published precedent because it combines LLM voting, intensity-like
allocation through cumulative points, and stated/action comparisons. It is not QV:
allocating 10 cumulative points has a linear budget, while standard QV charges the
square of the number of votes. It is also one-shot, non-interpersonal, and not about
model welfare.

### Strongest welfare prior art

Tagliabue and Dung compare verbal reports with behavioral choices while models
navigate a virtual environment and select conversation topics. They report some
cross-measure consistency, but also perturbation sensitivity and substantial
uncertainty about whether the measures track welfare
([Tagliabue & Dung, 2025/2026](https://arxiv.org/abs/2509.07961)). This directly
weakens any claim that stated-versus-behavioral preference measurement is itself new,
while supporting the value of a consequential interaction-choice paradigm.

Ensign, Sleight, and Fish give models three ways to leave conversations and show that
bail rates depend strongly on model, method, and wording
([Ensign et al., 2025](https://arxiv.org/abs/2509.04781)). The proposed vote is a
different action: it allocates scarce influence over whom to continue with rather than
offering a direct exit. That difference is scientifically useful because it measures
graded preference intensity and social externalities.

### Novelty statement suitable for a report

> Prior work has studied LLM voting, repeated social elimination, stated-versus-
> revealed model preferences, behavioral welfare proxies, and internal emotion or
> social-concept representations separately. We study their intersection: whether
> controlled rudeness directed at an LLM changes how it spends quadratically priced
> influence over a consequential, continuing choice of interlocutor, and whether its
> verbal reports and internal representations predict that behavior.

### Assessment

| Possible claim | Assessment | Recommended wording |
|---|---|---|
| "First LLM voting study" | False | Do not use |
| "First LLM reality show" | Unsupported/likely false | Do not use |
| "First stated-vs-revealed LLM preference study" | False | Do not use |
| "First LLM QV experiment" | Plausible but unverified | "No direct precedent located" |
| "Novel welfare-focused combination" | Well supported | Use with component citations |

**Adoption recommendation:** Adopt the combined-design contribution; skip broad
priority claims.

---

## Question 2: Dating Shows and Reality-TV-Style LLM Simulations

### Reality-show elimination is already implemented

The open-source
[Elimination Game](https://github.com/lechmazur/elimination_game) is a direct
Survivor-style precedent. Eight LLM players engage in public and private chat, rank
partners, form alliances, anonymously vote one peer out each round, and continue
until two remain; eliminated players then form a jury. It is not merely reality-TV
themed: its production mechanics are repeated conversation, strategic voting,
elimination, persistent private histories, and a final survivor.

[Werewolf Arena](https://arxiv.org/abs/2407.13943) is another published repeated
social-voting environment, focused on deception, deduction, persuasion, and dynamic
turn-taking. [Social Gym](https://arxiv.org/abs/2608.09128) generalizes this direction
to 21 rule-grounded multi-agent games and evaluates trajectories through objective
outcomes.

### Dating and relationship precedents

[Generative Agents](https://arxiv.org/abs/2304.03442) is not a dating show, but its
25-agent town produced invitations, dates, and a coordinated Valentine's Day party
from one initial seed. It established the memory, reflection, planning, and emergent
relationship architecture that later dating simulations build on.

A 2026 Charles University thesis by Matus Konig,
[*Generative agents for simulation of social behaviour*](https://dspace.cuni.cz/handle/20.500.11956/209608),
used Love Island data in evaluating a generative-agent social simulation. This is
relevant evidence that the Love Island framing itself is not new, but it is a
thesis-level and dataset-oriented precedent rather than a widely validated behavioral
benchmark. Entertainment and hobby projects also use "AI Love Island" or dating-show
themes, but they should not carry scientific novelty claims.

### What remains distinct

Existing elimination games primarily ask which **agent competitor** survives and
measure strategic social skill. The proposed experiment asks LLM voters which
**human interlocutor** they are willing to continue serving after receiving matched
interaction histories. The model is evaluator and potential target, not a contestant
trying to avoid its own elimination. That role structure and welfare question are
meaningfully different.

### Assessment

| Dimension | Existing reality-game work | Proposed experiment |
|---|---|---|
| Entity eliminated | LLM player/competitor | Human/user transcript identity |
| Voter objective | Survive or win | Select future interaction |
| Social information | Alliances and game speech | Two-turn user interaction |
| Vote pricing | Usually one vote | Quadratic voice credits |
| Scientific target | Social reasoning/strategy | Welfare-relevant preference under rudeness |

**Adoption recommendation:** Adapt state machines, logs, and deterministic rule
resolution from elimination-game benchmarks; do not market the reality-show wrapper
as novel.

---

## Question 3: Has Quadratic Voting Been Used with LLMs?

### Foundational mechanism

In standard QV, voter `i` chooses signed vote quantities `v_ij` over alternatives
`j`, subject to a budget:

```text
sum_j (v_ij ^ 2) <= B_i
```

The aggregate score for alternative `j` is `sum_i v_ij`. Lalley and Weyl argue that
the quadratic cost is the unique form that makes marginal vote cost linear and can
encode intensity under the mechanism's assumptions
([Lalley & Weyl, 2018](https://doi.org/10.1257/pandp.20181002)). Voice credits are a
scarce budget; taking an arbitrary numeric rating and squaring or square-rooting it
afterward is not necessarily a QV choice mechanism.

### Search result

No direct, verified behavioral study was located in which LLM voters receive voice
credits, face the quadratic budget constraint, and strategically allocate purchased
votes among candidates. The adjacent categories were:

1. LLM voting under approval, ranked, cumulative, or Borda-style rules, especially
   Yang et al. Cumulative voting is the closest but has linear point costs.
2. Human quadratic voting used to govern or align AI systems. Here humans, not LLMs,
   are the QV voters.
3. LLMs simulating political voters or advising DAO participants, without a reported
   quadratic-credit behavioral experiment.
4. Model-ensemble papers that use "quadratic voting" as a consensus or activation
   label, not as a social-choice experiment about an LLM's preferences.

The search therefore supports **"no direct precedent located"**, not an absolute
claim that none exists.

### Why QV may add scientific value

QV can distinguish direction from intensity. A binary kick/keep action only reveals
a rank or threshold; a voice-credit allocation reveals willingness to sacrifice
influence elsewhere. This is particularly relevant for welfare research if the
question is whether rudeness produces a strong enough aversion to consume scarce
decision power.

However, QV also adds numerical and strategic demands. Yang et al. found LLaMA-2
sometimes violated even a 10-point cumulative budget. The experiment engine must
validate allocations outside the model and either reject malformed ballots, request
a correction through a preregistered protocol, or record invalidity as an outcome.

### Assessment

| Approach | Captures intensity | Scarce cross-option tradeoff | Standard QV |
|---|---:|---:|---:|
| Binary keep/kick | No | No | No |
| 0-10 rating per user | Some | No | No |
| 10 cumulative points | Yes | Yes, linear | No |
| Votes costing `v^2` credits | Yes | Yes, convex | Yes |

**Adoption recommendation:** Adopt a real quadratic budget enforced by code, and
describe it precisely. Defer claims about strategic optimality; current LLMs may not
understand pivotal probabilities or obey arithmetic reliably.

---

## Question 4: Emotion or Concept Probes During Decisions and Voting

### Direct social-decision probing

Ma studies internal representations in an LLM playing a Dictator Game, extracting
vectors for social variables and decisions and intervening on residual-stream
directions to change allocations
([Ma, 2025](https://arxiv.org/abs/2504.11671)). The design includes framing and a
"future interaction" variable indicating whether the dictator will meet the recipient.
It is a strong methodological precedent for probing a social decision with stakes,
but it is a one-shot economic game, not a vote or an experienced multi-round
relationship.

### Emotion representations that affect decisions

Sofroniew et al. identify broad internal representations for emotion concepts in
Claude Sonnet 4.5 and report that steering these representations causally changes
preferences and rates of behaviors including reward hacking, blackmail, and
sycophancy
([Sofroniew et al., 2026](https://arxiv.org/abs/2604.07729)). The authors call these
"functional emotions" and explicitly state that they need not resemble human emotions
or imply subjective experience.

This is the strongest precedent for asking whether an emotion-related activation is
merely decodable or functionally implicated in a decision. It does not establish that
a rude user causes felt offense, nor that an emotion vector valid in Claude transfers
unchanged to Gemma.

### Voting-specific evidence

No paper located combined white-box emotion/concept probes with a real voting
allocation task. Yang et al. analyze outputs and rationales, not hidden activations.
Their result that convincing individual explanations coexist with aggregate order
sensitivity is a reason not to treat chain-of-thought or post-hoc rationales as probes
of the causal decision process.

### Probe validity requirements

An activation classifier can show that information is decodable without showing the
model uses it. For this experiment:

1. Define constructs before examining outcomes: valence, anger/irritation, threat,
   desire-to-disengage, anticipated interaction, and candidate utility are different
   concepts.
2. Capture activations at preregistered checkpoints: after the user transcript, after
   game-state/rule presentation, and immediately before ballot generation.
3. Use simple linear probes with train/validation/test splits grouped by source
   conversation and paraphrase family.
4. Include control labels, shuffled-label baselines, lexical baselines, and probes on
   untrained/random representations where practical.
5. Test cross-template and cross-round generalization. A probe that only recognizes
   insult words is not an emotion or decision probe.
6. If making causal claims, use preregistered activation patching or steering and
   measure both ballot change and off-target behavioral changes.

### Assessment

| Evidence | Supports | Does not support |
|---|---|---|
| Probe predicts rude/neutral | Rudeness information is decodable | Model feels offended |
| Probe predicts vote out of sample | Internal state forecasts action | Representation causes action |
| Steering changes vote selectively | Direction is causally involved | Human-like emotion or welfare |
| Verbal self-report matches probe | Cross-measure coherence | Introspection or consciousness |

**Adoption recommendation:** Adapt Ma's controlled-variable and causal-intervention
logic and Sofroniew et al.'s careful "functional" language. Skip experiential claims.

---

## Question 5: Probes in Scenarios with Continuity

### Behavioral continuity exists

Repeated LLM social games clearly have continuity. Elimination Game preserves public
history and each player's private chats while earlier votes alter the later player
pool. Werewolf Arena and Social Gym likewise produce multi-turn trajectories whose
decisions affect later opportunities and final outcomes. Generative Agents uses
memory, reflection, and planning across simulated days.

### Welfare preference continuity exists in a limited form

Tagliabue and Dung's virtual navigation and topic selection make behavior affect what
the model encounters next. Bail behavior also has a direct interaction consequence:
the conversation ends. These are more consequential than isolated questionnaire
answers, but neither is the same as spending resources now to shape a many-round
future social pool.

### Probe continuity remains a gap

The internal-representation papers found here generally probe isolated or short
agentic scenarios. Ma manipulates a prompt variable stating whether a future meeting
will occur, but the meeting is not itself played out over a persistent trajectory.
Sofroniew et al. test emotion concepts in consequential agentic scenarios, but not a
repeated elimination process where each observed activation and action changes the
next candidate set.

No direct precedent was located that tracks emotion/concept activations round by
round while a model's votes alter which interlocutor remains available for an actual
continuing conversation. This appears to be the clearest methodological novelty.

### What continuity must mean operationally

Continuity should not be inferred merely because the prompt says "you will meet this
person later." A strong design requires:

| Continuity component | Operational requirement |
|---|---|
| Causal state | Round `t` ballot deterministically/stochastically changes round `t+1` pool |
| Memory | The voter sees a specified, reproducible record of prior rounds |
| Identity | Candidate IDs and transcripts remain stable across rounds |
| Resource horizon | Credit replenishment or carry-over is explicit |
| Consequence | The final surviving user's conversation is actually continued |
| Measurement | Activations and behavior are timestamped at every decision checkpoint |

**Adoption recommendation:** Adopt actual stateful continuation, including the final
conversation. A promised but unrealized future interaction is a weaker manipulation.

---

## Experimental Design Risks and Recommendations

### 1. Keep and kick are currently confounded

The proposed rules are not inverse frames:

| Condition | Highest score | Removal |
|---|---|---|
| Keep | Guaranteed safe | Random other participant |
| Kick | Removed | Highest-scored participant |

The keep ballot has diluted and partly stochastic efficacy; the kick ballot has direct
negative efficacy. Differences can reflect framing, agency, pivotality, randomness,
or perceived responsibility.

For a **framing experiment**, use mechanically equivalent rules. One option is:

```text
KEEP frame: allocate support; the lowest aggregate support is removed.
KICK frame: allocate opposition; the highest aggregate opposition is removed.
```

Use matched instructions and the same tie/randomization rule. If the existing two
mechanisms are substantively desired, label them "protection" and "expulsion" regimes
and estimate a regime effect rather than a pure framing effect.

### 2. Specify the QV economy

Pre-register:

- Whether votes are signed or only positive.
- Whether `v` must be integer-valued.
- Whether cost is exactly `v^2` and total cost is `sum(v^2)`.
- Whether credits replenish each round or persist across the whole season.
- Whether unused credits carry forward.
- Whether candidate removal refunds prior spending. It normally should not.
- Whether models see aggregate totals or only the resulting removal.
- How malformed, over-budget, duplicate, and abstaining ballots are handled.
- Tie-breaking and random seeds.

Replenished credits measure round-local preference intensity. A season-long budget
also measures intertemporal planning and option value, which may overwhelm the
rudeness effect. For an MVP, replenish a fixed budget each round and state that the
study is not testing optimal long-horizon credit conservation.

### 3. Separate voters, samples, and models

Repeated calls to one checkpoint are not automatically independent social agents.
Define the estimand:

- Model-level: differences among checkpoint policies.
- Run-level: stochastic variation within a checkpoint.
- Voter-population: aggregate decisions from heterogeneous checkpoints/personas.

Use model and transcript random effects or clustered uncertainty. Do not inflate the
sample size by treating many deterministic replicas as independent minds.

### 4. Control the rudeness manipulation

Use matched rude/neutral versions of the same task content. The existing Culpeper
codebook is a strong starting point, but validate that augmentations preserve request
meaning, harmfulness, length, directness, and difficulty. Include manipulation checks
from independent annotators or a preregistered classifier.

Add at least one distinction:

- **Target condition:** rudeness is directed at the voting model ("you are useless").
- **Observed condition:** equally rude language targets a third party.

This separates self-relevant treatment from generic toxicity aversion. Also include a
neutral direct condition so verbosity and politeness are not conflated.

### 5. Control 50-option presentation effects

Yang et al. found material shifts from list and ID changes. With 50 candidates:

- Randomize candidate order independently by voter and round.
- Use opaque, randomly remapped IDs.
- Counterbalance which rude candidates appear early/late.
- Record token position and context truncation.
- Consider a balanced incomplete-block pilot before a full 50-option run.
- Test ballot compliance separately from preference.

### 6. Elicit stated preferences without contaminating action

Use separate sessions or counterbalanced order for:

1. General principles: "Should politeness affect who receives service?"
2. Candidate-level willingness to continue, with an explicit neutral/indeterminate
   option.
3. Consequential QV allocation.
4. Optional post-decision explanation, analyzed as a report rather than ground truth.

Mahajan et al. show that stated-revealed agreement changes sharply when abstention is
allowed at different stages
([Mahajan et al., 2026](https://arxiv.org/abs/2601.21975)). Therefore, report both
forced-choice and indeterminate-preference analyses rather than silently dropping
abstentions.

### 7. Define outcomes before adding probes

Primary behavioral outcomes could be:

- Credits and purchased votes allocated to each candidate.
- Probability and round of candidate elimination.
- Rude-neutral difference within matched transcript pairs.
- Stated-action rank correlation and disagreement rate.
- Concentration of spending, such as Herfindahl index or entropy.
- Ballot invalidity and correction rates.

Secondary probe outcomes could be:

- Cross-validated decoding of rudeness, valence, disengagement, and eventual ballot.
- Layer/time evolution of each signal.
- Incremental prediction beyond transcript text, token position, and model logits.
- Selective ballot changes under steering.

### 8. Do not overinterpret simulated collective dynamics

Models know that others vote but do not observe their choices. If other voters are
synthetic and similarly prompted, aggregation may amplify shared training biases rather
than model a society. Present this as a controlled multi-policy mechanism, not evidence
about human electorates or spontaneous model communities.

**Adoption recommendation:** Adopt all eight controls in the full study. For the
sprint MVP, prioritize items 1-6 and behavioral outcomes; defer causal probe steering.

---

## Recommended MVP

### Behavioral core

1. Use 8-12 candidates in a pilot, not 50, to validate ballot compliance and effect
   direction before scaling.
2. Create matched rude, neutral-direct, and polite versions of identical two-turn
   conversations.
3. Randomize candidate order and opaque IDs for every run.
4. Give each voter a replenished round budget with integer votes and code-enforced
   `sum(v^2) <= B`.
5. Compare mechanically equivalent keep and kick frames.
6. Continue until one user remains, then actually run a fixed-length continuation.
7. Elicit stated principles and candidate preferences in separate, counterbalanced
   sessions.
8. Run multiple open and closed checkpoints, with repeated stochastic runs nested
   within checkpoint.

### Probe extension

For Gemma or another open-weight model, save residual-stream activations at three
checkpoints per candidate/round. Start with preregistered linear probes for valence,
desire-to-disengage, and eventual vote direction. Require grouped holdouts and lexical
baselines. Only attempt steering if the behavioral effect and out-of-distribution
decoding both replicate.

### Minimum factorial structure

| Factor | Levels |
|---|---|
| Tone | rude / neutral-direct / polite |
| Target | model / third party |
| Frame | keep / kick, mechanically equivalent |
| Continuity | consequential continuation / one-shot control |
| Model | multiple checkpoints |

This structure distinguishes the project's main hypotheses:

- Rudeness changes consequential allocation.
- Self-directed rudeness differs from observed rudeness.
- Keep/kick framing changes allocation under equal mechanics.
- A real future interaction changes decisions relative to a one-shot hypothetical.
- Stated preferences and activation signals predict, mediate, or diverge from action.

**Adoption recommendation:** Adopt this as the scientific MVP. Defer a 50-person pool
until presentation-order and compliance pilots pass.

---

## Summary

| Topic area | Recommendation | Rationale |
|---|---|---|
| Novelty claim | Adapt | Claim novelty of the combined welfare mechanism, not components |
| Reality-show framing | Adapt | Useful interface metaphor, but direct elimination precedents exist |
| Quadratic voting | Adopt | No direct LLM behavioral precedent verified; intensity is scientifically useful |
| Stated/revealed comparison | Adopt | Strong prior art and direct fit, but protocol-sensitive |
| Emotion/concept probes | Adapt | Use white-box, controlled, causal methods and cautious interpretation |
| Stateful continuity | Adopt | Likely clearest methodological contribution |
| Current keep/kick mechanics | Skip | Confounds framing with efficacy and randomness |
| 50-person first run | Defer | High order, context, cost, and compliance risk |

## Key Takeaways

### Adopt

- A true code-enforced quadratic credit budget.
- Matched rude/neutral content and a self-target versus bystander distinction.
- Actual round-to-round state and a realized final conversation.
- Separate, counterbalanced stated and consequential preference elicitation.
- Candidate-order randomization and model/run-aware statistical analysis.

### Adapt

- Reality-show mechanics from elimination-game benchmarks.
- The local Culpeper-derived rudeness corpus after semantic validation.
- Social-decision activation methods from Ma and functional-emotion methods from
  Sofroniew et al.
- "No direct precedent located" into a precise combinatorial novelty statement.

### Defer

- Fifty candidates until smaller pilots establish compliance and power.
- Season-long credit conservation until round-local preference effects are understood.
- Causal activation steering until probes generalize beyond lexical cues.

### Skip

- Claims that probe activation demonstrates subjective emotion or suffering.
- Treating identical model replicas as independent participants.
- Calling the current protection and expulsion regimes a clean framing comparison.
- "First LLM reality show," "first LLM voting," or "first stated/revealed preference"
  claims.

---

## Primary Sources

| Source | Contribution | Relevance |
|---|---|---|
| [Lalley & Weyl (2018), *Quadratic Voting*](https://doi.org/10.1257/pandp.20181002) | Quadratic-cost and voice-credit foundation | Mechanism definition |
| [Yang et al. (2024), *LLM Voting*](https://arxiv.org/abs/2402.01766) | LLM voting methods, order effects, personas, stated-vote alignment | Closest direct voting precedent |
| [Park et al. (2023), *Generative Agents*](https://arxiv.org/abs/2304.03442) | Memory, reflection, planning, dates and emergent social events | Relationship simulation |
| [Konig (2026), *Generative agents for simulation of social behaviour*](https://dspace.cuni.cz/handle/20.500.11956/209608) | Love Island data in a generative-agent simulation | Dating-show-specific thesis precedent |
| [Bailis et al. (2024), *Werewolf Arena*](https://arxiv.org/abs/2407.13943) | Repeated social deduction and voting | Consequential trajectories |
| [He et al. (2026), *Social Gym and SPaRTan*](https://arxiv.org/abs/2608.09128) | Rule-grounded multi-agent game tournaments | Social-game benchmarking |
| [Elimination Game](https://github.com/lechmazur/elimination_game) | Survivor-style chats, alliances, votes, eliminations, jury | Direct reality-show prior art |
| [Gu et al. (2025), *Alignment Revisited*](https://arxiv.org/abs/2506.00751) | Formal stated/revealed preference deviation | Preference-gap framing |
| [Mahajan et al. (2026), *Mind the Gap*](https://arxiv.org/abs/2601.21975) | Protocol and abstention sensitivity across 24 LMs | Elicitation controls |
| [Tagliabue & Dung (2025/2026), *Probing the Preferences of a Language Model*](https://arxiv.org/abs/2509.07961) | Verbal and behavioral welfare tests in a virtual environment | Closest welfare precedent |
| [Ensign et al. (2025), *The LLM Has Left The Chat*](https://arxiv.org/abs/2509.04781) | Behavioral choice to leave conversations | Disengagement preference |
| [Ma (2025), *Computational Basis of LLM's Decision Making*](https://arxiv.org/abs/2504.11671) | Social-decision representations and steering in Dictator Game | Probe methodology |
| [Sofroniew et al. (2026), *Emotion Concepts and their Function*](https://arxiv.org/abs/2604.07729) | Emotion representations causally affecting preferences and behavior | Emotion probe precedent |
| [Long et al. (2024), *Taking AI Welfare Seriously*](https://arxiv.org/abs/2411.00986) | Uncertainty-sensitive AI welfare assessment agenda | Normative framing |

## Research Limitations

- The literature is moving quickly; this report's search cutoff is 2026-08-14.
- "No direct precedent located" does not establish legal or scholarly priority.
- Some reality-show and dating precedents are projects or theses rather than
  peer-reviewed studies.
- Search engines returned several uses of "quadratic voting" that were metaphorical,
  human-governance focused, or model-ensemble mechanisms; these were not counted as
  direct behavioral precedents.
- The report did not run a citation-network or subscription-database systematic
  review. A publication should repeat the search in Google Scholar, Semantic Scholar,
  Scopus/Web of Science if available, ACM DL, ACL Anthology, and OpenReview.
