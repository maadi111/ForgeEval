# Architecture

```text
                         ForgeEval
                            |
             +--------------+--------------+
             |                             |
        Task Registry                 Submission API
             |                             |
        Task Package                  Run Manager
             |                             |
       +-----+------+              Docker Executor
       |            |                     |
 Public Tests   Hidden Tests          Candidate
       |            |                     |
       +-----+------+---------------------+
             |
          Grader
             |
   +---------+----------+
   |                    |
Correctness        Adversarial
Regression         Performance
Generalization    Integrity
   |                    |
   +---------+----------+
             |
        Score + Report
             |
       PostgreSQL/MLflow
             |
         Dashboard
```

## Trust boundary

Candidate code is untrusted. It runs inside a constrained Docker sandbox. Hidden tests and grader logic must remain outside the candidate-controlled workspace.

## Design rule

The grader evaluates contracts and observable behavior, not implementation identity.
