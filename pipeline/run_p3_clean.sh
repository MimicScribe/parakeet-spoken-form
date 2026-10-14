#!/bin/bash
for l in de es fr it nl pt; do modal volume rm parakeet-spoken-form ml/p3/$l/wav -r >/dev/null 2>&1; modal volume rm parakeet-spoken-form ml/p3/$l/rows.jsonl >/dev/null 2>&1; done
modal volume ls parakeet-spoken-form ml/p3/de
