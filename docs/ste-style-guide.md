# The writing standard: ASD-STE100 Simplified Technical English for eta-eats

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **raw file** | A Kaggle CSV file before cleaning: `train.csv` or `test.csv`. | dataset (alone), input data |
| **delivery** | One row of a raw file: one order from a restaurant to a customer. | order (for a row), trip, record |
| **courier** | The person who delivers, with a courier ID such as `INDORES13DEL02`. | driver, rider, delivery person (in prose) |
| **courier city** | The city code at the start of the courier ID, for example `INDO`. | region, hub |
| **delivery time** | The target: minutes from the order to the delivery (`time_taken_min`). | ETA (in prose), duration, label |
| **traffic level** | The ordered value of the traffic density: Low 0, Medium 1, High 2, Jam 3. | traffic score, congestion |
| **category map** | The fixed set of known values for one categorical column. | mapping table, dictionary |
| **data quality report** | The counts that the cleaning step prints: missing values, fixes and roll-overs. | log, summary |
| **roll-over** | A pick-up time that is earlier than the order time, moved to the next day. | wrap, overflow |
| **preparation time** | Minutes from the order time to the pick-up time (`prep_min`). | prep (in prose), wait time |
| **distance** | The Haversine distance in kilometres from the restaurant to the customer. | range, length |
| **interaction** | The feature `traffic_x_distance`: traffic level multiplied by distance. | cross feature, product term |
| **row-wise feature** | A feature that uses the values of one row only. It has no fitted state. | engineered feature (alone) |
| **pipeline** | The sklearn `Pipeline`: preprocessor plus regressor, fitted on training rows only. | workflow, flow |
| **preprocessor** | The `ColumnTransformer` for imputation, scaling and one-hot encoding. | transformer (alone), encoder |
| **split** | The division into training rows and test rows: `time`, `courier` or `random`. | partition, fold (for the test part) |
| **fold** | One part of the courier-grouped cross-validation on the training rows. | split (for a CV part) |
| **baseline** | The `median` model, which predicts the median delivery time. | dummy model, benchmark |
| **tuned model** | The refitted best pipeline of the randomized search. | final model (alone), optimized model |
| **slice** | A group of test rows with one value of one column, for example traffic `Jam`. | segment, cohort |
| **permutation importance** | The increase of the test MAE when the values of one input column are shuffled. | feature importance (alone), weight |
| **model card** | The file `model_card.md` with the intended use, data, metrics and limits. | report, documentation |
| **submission** | The CSV file with `ID` and the predicted `Time_taken (min)` for each test delivery. | output file, results |
| **synthetic data** | The generated files of `eta-eats synth`. They are not real deliveries. | fake data, mock data, sample data |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **load** | Read a raw file, clean it and add the row-wise features. |
| **clean** | Remove spaces, change `NaN` strings to missing values and check every category. |
| **map** | Change a category to its ordered number with the category map. |
| **roll over** | Add one day to a pick-up time that is earlier than its order time. |
| **split** | Divide the rows into training rows and test rows. |
| **fit** | Train the pipeline on the training rows. |
| **tune** | Search the parameters of the `gbm` pipeline with courier-grouped folds. |
| **score** | Calculate the metrics on the test rows. |
| **predict** | Calculate delivery times for a file without a target. |
