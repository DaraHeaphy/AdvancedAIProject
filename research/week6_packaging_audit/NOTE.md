# Preliminary packaging audit — excluded from the main comparison

While checking portability, the new comparison tests were found to import
`comparison.py`, which the original evidence source list did not include.
The six training runs finished before the runner was stopped. The packager was
corrected to retain that script and its plotting requirements, then the complete
comparison was restarted under a consistent source snapshot in `research/week6/`.

These preliminary runs are retained for transparency. They use the same seeds
and are not independent extra evidence. They are not counted in the main Week 6
results table. Their exclusion is because of the packaging correction, not their
metric values. The final report accounts for this additional training compute.
