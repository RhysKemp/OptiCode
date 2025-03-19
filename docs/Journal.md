# OptiCode Journal & Notes

## Scope

- **Types of Code:** General purpose code optimisation.

## Optimisations Priority

1. Execution speed
2. Memory usage
3. Energy consumption
4. Readability/Maintainability

## AI Techniques Utilised

- Most logical

## How the Tool Will Operate

- Initially automatic.
- Open to future integration.

## Model Training

### Datasets

- Datasets from GitHub (require before and after optimisation examples)
- Can generate synthetic data, to be posted in a GitHub repo
- **Optimisation Examples Repo:** [TODO LINK](#)
- **Optimisation Model Training dataset** [LINK](https://github.com/CodeGeneration2/Efficient-Code-Generation-with-GEC)

### Other Examples

- TODO
- Model training will likely be a mix of **Reinforcement Learning (RL)** and **Supervised Learning (SL)**.
- Initially, build a modular tool that uses a pre-trained model as a starting point.

## Benchmarking

- **Status:** TODO
- Need to determine how to measure benchmark stats and what telemetry to utilise.
- This will be addressed after creating the initial framework and test case.

## Architecture

- **Research Status:** TODO

### Input Handling

- Prototype will simply take raw source code as input.
- Initial focus on Python; using Abstract Syntax Trees (ASTs) can keep it language neutral.

### Optimisation Pipeline

- **Key Questions:**

  - How will we analyse the code?

    - Tokenise and then perform AST parsing

  - What are the pros and cons of methodologies such as AST parsing, tokenization, and graph representations?

    - Tokenisation is fast but lacks and deep understanding of structure.
    - AST parsing provides a more detailed and nuanced understanding of the structure of the code, but is more computationally expensive than just tokenisation.
    - GRAPHS TODO

  - How will the tool integrate with the pipeline to suggest or apply improvements?

    - Accepts raw source as an input
    - Converts into tokens or generates the AST
    - AI model processes the tokenised code or AST to identify optimisations.
    - Based on suggestions the tool applies or suggests the optimisations.
    - Optimised code is outputted.

  - Will there be a post-processing step to validate optimizations?

    - There will initially be another tool that can be run on the optimised code, later it will be automatically run.
    - The tool will measure the specific benchmarks previously listed before and after the transformation.
    - Potential static analysis to ensure new code adheres to codign standards and maintains code integrity.
    - Much later in the lifecycle of the project, user feedback could be implemented.

### Output & User Interaction

- Will the tool automatically rewrite the code and provide explanations?

  - The tool will initially in the prototype produce a second file of code with the optimisations applied and perhaps another file with rationale could be implemented.

- Will it generate reports on improvements (speed, memory, energy)?

  - Initially in the prototype, we will use a second tool to measure and analyse performance metrics and compare manually.
  - Later we can implement automatic comparison and even UI.

- How will users evaluate changes (e.g., diff comparisons, benchmarks)?

  - Diff comparisons, explanations and benchmarking will be the primary methods of evaluation.

## Roadmap

- Phase 1:
  - [x] Step 1: Set up code parsing (AST and tokenization).
  - [ ] Step 2: Implement basic rule-based optimisations.
  - [ ] Step 3: Implement benchmarking tool.
- Phase 2:
  - [ ] Step 4: Research existing AI models for code optimisation
  - [ ] Step 5: Collect and prepare data.
  - [ ] Step 6: Train and evaluate an inital model.
- Phase 3:
  - [ ] Step 7: Integrate the AI Model into the pipeline
  - [ ] Step 8: Continuous improvements.

## Journal

### Day 1

- Defined project scope.
- Began thinking about code collection.
- Considered using a pre-trained model such as GPT-3.
- Noted that benchmarking will need completion at a later stage, with current priorities outlined.
- Started preliminary architecture planning; further research required.

### Day 2

- Begin ideation on the actual contruction of the tool.
- Work on the timeline for the next few weeks of coding.
- Answered the remaining questions about output and the optimisation pipeline.
- Created project file structure

### Day 3

- Create tokenization class and unit test

### Day 4

- Created [ast_parser.py](../engine/ast_parser.py) - A simple ast parser and visualiser class
- Found a [DATASET](https://github.com/CodeGeneration2/Efficient-Code-Generation-with-GEC) for inefficient-efficient code pairs
- Set up [ci.yml](../.github/workflows/ci.yml) workflow for unit testing
- TODO test_ast_parser.py - ast parser unit tests

### Day 5

- Create [test_ast_parser.py](../tests/test_engine/test_ast_parser.py) - unit tests for [ast_parser.py](../engine/ast_parser.py)
- Added various methods to [ast_parser.py](../engine/ast_parser.py) for manipulating ASTs
- Started work on rule-based optimisations, currently only un-used variables. Framework should allow for other rules to be implemented fairly easily.
- Unit tests need to be expanded for both referenced classes.

### Day 6

- Add more unit testing for [ast_parser.py](../engine/ast_parser.py)
- Begin work on [rule_based_optimisations.py](../engine/rule_based_optimisations.py)

### Day 7

- Continue work on [ast_parser.py](../engine/ast_parser.py) and [rule_based_optimisations.py](../engine/rule_based_optimisations.py)
- Add some work on [benchmark.py](../benchmarking/benchmark.py)

### Day 8

- Finished implmenting [benchmark.py](../benchmarking/benchmark.py) and created [test_benchmark.py](../tests/test_benchmarking/test_benchmark.py)
- Begin work on AI integration.
- Start creating a [dataset](LINK TODO) class to load in the open source GEC dataset (MIT open source license)
