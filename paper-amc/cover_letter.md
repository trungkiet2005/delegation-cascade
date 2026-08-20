# Cover letter

*Replace the bracketed fields before submitting. Paste the body into the
"Enter Comments" box of the Editorial Manager submission, or upload it as a
separate "Cover Letter" item.*

---

[Date]

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
   strictly lowers the liability charged to the principal and strictly raises
   the harm it causes (Theorem 4). We prove the converse condition: an
   attribution rule that is bounded below removes the effect (Proposition 8).
   Two further exact statements order the placement of a verification step
   along the chain and establish monotonicity of harm in depth.

2. **Computation.** The joint space of depths and specifications makes the
   mutation-selection chain over population states a Markov chain on
   3.0 x 10^27 states at the parameters used, so it cannot be formed. We give a
   scheme that avoids it: the interaction tensor is summed exactly over the
   horizon law rather than sampled, the process is reduced to an embedded chain
   on the pure designs, and the fixation probabilities that chain requires are
   evaluated from a closed form in O(Z) operations with the largest exponent
   factored out of a sum whose terms overflow double precision by three orders
   of magnitude in the exponent. One stationary regime costs 28 ms; the full
   21 x 21 parameter plane costs under a minute. Section 5.6 verifies the
   scheme against a 60-digit reference, against an independent implementation
   and against the full mutation-selection chain on a design space small enough
   to carry one.

3. **Numerical results.** The two per-hand-off losses raise the long-run unsafe
   frequency by less than 0.02 each and to 0.32 together, so 91% of the effect
   is interaction rather than either mechanism. A statutory floor under
   attributed responsibility traces the same safety and welfare frontier as a
   cap on chain length, with a mean gap of 0.005 at matched welfare. Improving
   transmission fidelity is not monotone in its own strength.

The manuscript is 25 pages in the Elsevier CAS single-column layout, including
figures, tables and references, and is therefore within the length the journal
asks authors to justify.

The work has not been published previously and is not under consideration
elsewhere. All authors have approved the submission. The code, the generated
tables, the numerical benchmarks and the figure pipeline are openly available
under the MIT licence at <https://github.com/trungkiet2005/delegation-cascade>,
so every number in the paper can be regenerated from the repository.

We have no competing interests to declare. We suggest the following reviewers,
none of whom has been involved in the work: [names, affiliations, emails].

Thank you for considering the manuscript.

Yours sincerely,

Trung-Kiet Huynh, on behalf of all three authors
Faculty of Information Technology, University of Science,
Vietnam National University Ho Chi Minh City, Vietnam
23122039@student.hcmus.edu.vn
