# Cover letter

*Paste the body into the "Enter Comments" box of the Editorial Manager
submission, or upload it as a separate "Cover Letter" item. Suggested reviewers
go in Editorial Manager's own field; see the note near the end.*

---

21 August 2026

To the Editors
*Applied Mathematics and Computation*

Dear Editors,

We are submitting the manuscript **"Delegation cascades: exact stationary
analysis of an evolutionary game with strategic chain depth"** for
consideration as a research article in *Applied Mathematics and Computation*.

The paper studies an evolutionary game in which the number of intermediaries
between a decision maker and the action it produces is itself a strategic
variable, and it is a computational paper about that game as much as a
structural one. Its content falls squarely in the journal's scope: it applies
systems-oriented ideas, Markov chains and evolutionary dynamics, to a
behavioural and social science question, and its emphasis is on the algorithm
that makes the model tractable, on the analysis of that algorithm, and on the
numerical results it produces.

The specific contributions are the following.

1. **Structure.** We prove that realised harm saturates in the delegation depth
   while attributed harm decays geometrically, so that for every attribution
   retention below one there is a depth beyond which each further hand-off
   strictly lowers the liability charged to the principal and never lowers the
   harm it causes (Theorem 4). We prove the converse condition: an attribution
   rule bounded below by a positive constant removes the effect
   (Proposition 8). Four further exact statements order the erosion sequence as
   a harm order in each argument, give the erosion law and its spectral rate,
   establish monotonicity of harm in depth, and order the placement of a
   verification step along the chain. All eight numbered results carry
   hypotheses and proofs.

2. **Computation.** The joint space of depths and specifications makes the
   mutation-selection chain over population states a Markov chain on
   3.0 x 10^27 states at the parameters used, so it cannot be formed. We give a
   scheme that avoids it: the interaction tensor is summed exactly over the
   horizon law rather than sampled, the process is reduced to an embedded chain
   on the pure designs, and the fixation probabilities that chain requires are
   evaluated from a closed form in O(Z) operations with the largest exponent
   factored out of a sum whose terms overflow double precision by three orders
   of magnitude in the exponent. The analysis that the size of this design space
   forces is not of the relative error but of the exponent range: what the
   embedded chain needs of its fixation probabilities is positivity, not
   accuracy, and Proposition 9 states the condition under which double precision
   still determines the stationary regime, which we then certify on all 462
   regimes the paper reports. We also give an a priori bound on the horizon
   truncation. One stationary regime costs 39 ms and the full 21 x 21 parameter
   plane 17.6 s, on hardware named in the paper. Section 5.6 sets out four
   checks of four different things: the interaction tensor against Monte Carlo,
   the arithmetic of the stabilised sum against a 60-digit reference, the closed
   form against an independent implementation, and the small-mutation reduction
   against the full chain.

3. **Numerical results.** From a calibration whose depth-zero unsafe frequency
   is 0.000, each per-hand-off loss acting alone leaves that frequency at or
   below 0.018 while the two together raise it to 0.322, so 91% of the effect is
   interaction rather than either mechanism. We report how that share moves with
   the two calibration constants, the liability level and the depth ceiling,
   rather than the median alone. A statutory floor under attributed
   responsibility reproduces the frontier of a depth cap wherever a floor can
   reach it, matching the unsafe frequency of each reachable ceiling to 0.0011.
   Improving transmission fidelity is not monotone in its own strength.

The manuscript is 29 numbered pages in the Elsevier CAS single-column layout,
including figures, tables and the reference list, and we should say plainly why
it is over the 25 pages beyond which the journal scrutinises length. Four of
those pages are the proofs and the two error bounds: every one of the nine
numbered results is stated with its hypotheses and proved, the horizon
truncation carries an a priori bound, and the floating-point condition the
stationary regime depends on is stated as a proposition and certified rather
than asserted. Three further pages are the reference list. We would rather
submit the proofs than the assertions, but we will cut Section 7 and the
related-work section further if the editors prefer a shorter paper.

The work has not been published previously and is not under consideration
elsewhere. All authors have approved the submission. The code, the generated
tables, the numerical benchmarks and the figure pipeline are openly available
under the MIT licence at <https://github.com/trungkiet2005/delegation-cascade>,
so every number in the paper can be regenerated from the repository.

We have no competing interests to declare.

<!-- TO COMPLETE BEFORE SUBMISSION: enter three or four suggested reviewers in
     Editorial Manager, none of them a co-author, a collaborator within the
     last three years, or a colleague at HCMUS. -->

Thank you for considering the manuscript.

Yours sincerely,

Trung-Kiet Huynh, on behalf of all three authors
Faculty of Information Technology, University of Science (HCMUS)
Vietnam National University -- Ho Chi Minh City (VNU-HCM)
Ho Chi Minh City, Vietnam
23122039@student.hcmus.edu.vn
