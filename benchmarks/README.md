# Benchmark Package Format

Each benchmark should eventually contain:

```text
task.yaml
Dockerfile
README.md
workspace/
data/
tests/public/
tests/hidden/
grader/
reference/solution-a/
reference/solution-b/
```

Hidden tests must not be exposed through the candidate workspace in a production implementation.
