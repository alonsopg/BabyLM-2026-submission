# Camera-Ready Review Fixes

This revision addresses the two official reviews of the BabyLM 2026 multi-task paper.

## Reviewer nrcs

- Added a dedicated Related Work section.
- Added the requested BabyLM 2025 multi-task reference: Kamzela, Lango, and Dusek (2025).
- Added a terminology paragraph before objective generation so RTD, connective examples, definiteness examples, semantic cloze, and relational cloze are defined before technical generation details.
- Renamed the generation subsection to "Objective inventory and generation" and reorganized it so the task inventory and terminology appear before generation mechanics.
- Expanded the related-work discussion of Kamzela, Lango, and Dusek (2025) to explain the closest BabyLM multi-task connection and how this paper differs.
- Added an explicit clarification that "multi-task" refers to alternating training updates into one shared encoder, not to ensembling or changing the submitted evaluation interface.
- Kept the paper framed as a mixed-result diagnostic study rather than as a uniformly positive multi-task method.

## Reviewer o3AD

- Added an explicit rationale for the objective mixture probabilities.
- Clarified that the probabilities are fixed pilot allocations, not tuned hyperparameters.
- Explained why the trainer samples one task per step.
- Added the limitation that one-task-per-step sampling may increase gradient variance compared with per-batch loss mixtures.
- Added the key confound that auxiliary effects may come from the objective, reduced MLM exposure, or their interaction.
- Reorganized the objective discussion around three loss interfaces: selective MLM, classification, and ranking.
- Added a loss-interface column to the objective-family table so the large objective inventory reads as a taxonomy rather than an unstructured list.
- Added an Analysis paragraph separating objective content, MLM displacement, and loss-interface effects.
- Added an enumerated key-findings list in the Results section for cleaner takeaways.
- Added a Limitations section that directly discusses single-seed comparisons, fast-evaluation interpretation, objective-weight schedules, and MLM-exposure displacement.
- Revised the Limitations section to frame these points as controlled scope boundaries and next-step design choices, rather than as a list of disqualifying weaknesses.

## Build Status

- Rebuilt `paper/source/babylm25-multitask-repair.pdf`.
- Synced the rebuilt PDF to `paper/babylm25-multitask-repair.pdf`.
- Final LaTeX log has no undefined citations, no unresolved references, and no overfull boxes.

## Figure Revision

- Merged the original thesis diagram and experimental-setup diagram into a single compact Figure 1.
- Removed the wide setup figure and redirected its discussion to the combined Figure 1.
- The new Figure 1 summarizes the fixed controls, varied objective interfaces, shared multi-task training, and diagnostic trade-off readout in one ACL-column-width diagram.
- Promoted the cleaner multi-task fusion diagram into the introduction as the main Figure 1.
- Removed the former Appendix B, Appendix C, and Appendix D material after the useful diagram content was folded into Figure 1.

## Table Revision

- Added grayscale row striping to the main results, delta, and best-by-metric tables.
- Marked improvements over the MLM-only baseline in bold black, preserving readability in grayscale print.
- Marked the MLM-only BLiMP score in bold black because it is the strongest observed grammar score.
- Removed the redundant best-by-metric table from the main body during the final format pass; the main and delta tables now carry the same comparison signal more compactly.

## Artifact Links

- Added a paper footnote pointing to the public code repository.
- Added a paper footnote pointing to the primary Hugging Face model artifact and additional rerun artifact.

## Contribution Framing

- Added "objective-interface alignment" as the named central takeaway in the abstract, introduction, results, analysis, and conclusion.
- Linked the semantic-cloze result explicitly to that takeaway: it uses the MLM head and a masked-token plausibility contrast, matching the EWoK-style scoring interface more closely than broad plausibility classification.

## ACL Format Check

- Ran the official ACL pubcheck tool with `python3 -m aclpubcheck --paper_type long paper/source/babylm25-multitask-repair.pdf`.
- The final rebuilt PDF passes with `All Clear!`.
- Compressed benchmark descriptions and repeated results prose so references begin on page 9, satisfying the long-paper main-text page limit.
