# Trust and Recipes — Plan

*Status: in progress, October 2026.*

Phase 1 is finished. Phase 2 is finished but has design mistakes, and Phase R fixes them before Phase 3 starts.

## Target state
This is what Womblex does once every item in this plan is done. Each requirement below is the whole requirement as it stands at the end of the plan, in the form the requirements document (functional_requirements.md) uses. Requirements this plan does not touch are not repeated, and gaps recorded in the requirements document that this plan does not close stay there. The requirements document's Outstanding list has no entry for Requirement 23 or Requirement 27.

### 2. Scale-Out and Environment-Agnostic Execution

**As** a user,
**I want** to spread processing across cloud machines and move steps between my own machine and the cloud,
**so that** I get more throughput without rewriting jobs or reading documents again.

**Given** cloud file storage and a queue database (Postgres, or a local SQLite file)
**When** the operator queues work and starts more cloud machines
**Then** the machines take groups of documents at the same time without doubling up, write the standard set of files, and write the run's document list when the steps finish.

**Acceptance criteria:**

- Cloud machines and local runs process a group of documents with the same code and write the same files in the same layout.
- Work runs through the queue, so a group is never processed twice, work that finished is not run again, and machines can be added or removed at any time.
- A cloud run's files can be copied to a local machine and read unchanged by the local step commands.
- A step runs directly over cloud storage under the same rules as over local files.
- Local and cloud storage addresses both work.
- Writing a step's results to cloud storage is all or nothing. Every output exists before any is published. A set left incomplete by a failed upload never looks complete, so the next run redoes that group and overwrites it. A step that replaces its own results does so in one action, so an interrupted publish leaves the previous results intact.
- When a cloud run's steps finish, Womblex writes the run's document list by itself. The command that writes it by hand still works, and running it again changes nothing.

### 12. Knowledge Graph and External Enrichment

**As** a user,
**I want** named people, organisations and their relationships found by an outside service and joined into one graph,
**so that** every mention points to the piece of text it appears in, whatever order the steps ran in.

**Given** text split into pieces and the outside service's credentials
**When** the operator runs the steps that find names and build the graph
**Then** names are found, retrying when the service limits requests, and a graph links each name to the pieces of text it appears in, without changing what earlier steps wrote.

**Acceptance criteria:**

- The step that finds names calls the configured service (by default Kanon-2 through Isaacus, retrying when the service limits requests). Only a service that returns a full document keeps it for reuse when text is split using AI. With any other service, text is split on Womblex's own machines and no paid call is made in its place.
- Building the graph produces its points and connections, and a separate set of links from each mention to the piece of text it appears in.
- The step that re-links mentions to pieces of text saves its links in a file of its own, marked with the exact version of the pieces it linked to. It never changes the files earlier steps wrote.
- Anyone reading the results can tell when the links no longer match the current pieces of text, and the matching links are used wherever they exist.
- Runs made before re-linking had a file of its own are read exactly as before.
- Re-linking gives the same result each time it runs, and never skips its work only because its file already exists.

### 14. Batch Processing, Resiliency, and Operational Controls

**As** a user,
**I want** reliable groups of documents, saved progress, and separate controls for each step from the command line and the web console,
**so that** I can manage large runs, recover from failures and inspect results easily.

**Given** a configured run, on a local machine or in the cloud
**When** the operator uses the command line or the web console
**Then** documents are processed in separate groups that can resume, each step can be started on its own, and results can be inspected.

**Acceptance criteria:**

- Documents are processed in groups of a set size. Results are added and progress is saved as each group finishes.
- Resuming an interrupted run picks up from the saved progress and skips documents already done.
- An error in one document stays with that document. It is recorded in the document list and does not stop the rest of its group.
- Each step can be started on its own from the command line. Queueing the same work twice changes nothing, so steps that depend on each other run in the right order.
- The retry command retries a failed cloud step from its first failed piece of work, so pieces that finished are not run again.
- A retry is a new start. It is refused before anything starts if the step is unknown, still running, not failed, or belongs to another client's run, and for any reason a new run would be refused: its settings, its AI models, no running machine checked for it, or inputs missing from storage.
- A step that ran after a failure ran on incomplete results. Once the failed step's latest try has succeeded, that later step can be retried too, in the order the operator runs them. Its retry redoes only the groups of documents the failure affected, and work that finished correctly is not sent to a paid service again.
- Logs can be written as one JSON record per line, with no extra software. Records from cloud machines and groups carry the run, job, step and document where known, and the default text log is unchanged.

### 16. Dashboard — Queue and Stage Progress

**As** an administrator,
**I want** a dashboard of the queue and each step's progress for one run,
**so that** I can watch throughput and spot stuck jobs while running or debugging a run, without touching the queue.

**Given** a chosen run, with an optional queue and the run's own saved progress for each step
**When** the operator opens the dashboard
**Then** it shows the run's status sentence, the queue counts and each step's progress, refreshing while the browser tab is visible and pausing while it is hidden.

**Acceptance criteria:**

- Queue and step state come from what the pipeline already writes: the queue and each step's saved progress. With no queue set up, the dashboard uses the saved progress.
- A run's status is shown as one plain sentence, built from the exact counts of work done, waiting and failed at each step. The counts are shown beside it.
- The sentence uses the same status words as the web API (pending, running, done, failed and cancelled), plus a warning when a scheduled run could not start its steps.
- The command line, the web API and the dashboard all show the same sentence for the same run.
- Job status uses the exact words the system writes (pending, running, done, failed and cancelled), plus stale for running work with no update past the staleness limit, and skipped.
- A retried step counts once. The run's counts use its latest try, and the try a retry replaced stays listed as failed, naming the retry that followed it.
- The run's history record, which other tools use to track where data came from, reports the run's real outcome: finished, failed or cancelled.
- The dashboard only names a stuck job. The queue recovers it when its machine restarts, and the dashboard never re-queues, cancels or takes work itself.
- Refreshing pauses while the browser tab is hidden, so a console left in the background stops querying the queue.
- Each step's progress is shown in the pipeline's declared order.
- Logs for the run's groups and steps are listed newest first, and each can be opened on its own.

### 22. Service API — Submission and Retrieval

**As** a developer connecting another system,
**I want** a web API, signed in by token, to upload documents, submit runs and read their results,
**so that** my software can use Womblex as a service without access to the machine it runs on.

**Given** the web service connected to storage, the queue and a document upload location, with a list of clients
**When** a caller presents a service token, uploads documents, submits a run and checks on it
**Then** the run is queued as the caller's and uses the settings the caller sent, and its status, files and text can be read only by that caller or an administrator, limited by permission and by how sensitive each file is.

**Acceptance criteria:**

- Every endpoint except health and readiness needs a bearer token whose SHA-256 fingerprint is in the client list. A missing or unknown token is refused with 401, and tokens are compared in constant time.
- Permissions are submit, read, read_raw and admin, where admin includes the rest. A request outside the caller's permissions is refused with 403.
- The service does not start with an empty client list unless run with `--insecure-no-auth`, and listens only on the local machine by default.
- An upload is written to `<ingest>/<client_id>/<upload_id>/`, keeping only bare file names. An unusable, repeated or unsupported name is refused with 400 and nothing is written, an over-limit request is refused with 413, and the response names the folder to submit.
- A client that is not an administrator submits runs only over its own folder (403 otherwise). The server makes each run's identifier, and it is unique even when submissions arrive at the same time.
- Submitted settings, a named preset or settings of the caller's own, are checked before anything is queued (400 on failure). Every part of the run uses them: they control how each step works, not only which steps run. Settings that turn on no later steps give a run that only reads the text.
- A run belongs to the client that submitted it. Another client's run is reported as not found (404), an administrator sees every run, and runs started from the command line, the web console or a workflow have no owner and are visible to administrators only. A run never has more than one owner.
- Document text comes from the version with personal details hidden by default. The original text layers need read_raw, decided by how sensitive each layer is under the data contract.
- A run's file listing gives each file's row count, contract version and sensitivity, so a caller can fetch the Parquet files directly under the data contract.
- A run's status is the same one-sentence status the command line and the dashboard show, with the counts beside it.
- A run started from a workflow that names a web address sends its notices to that address. Callers can still check a run's status themselves, and a caller with no web address does so.
- Readiness reports whether storage and the queue can be reached, and answers 503 when either is down. Every other endpoint answers 503, not 500, when the queue, storage or upload location cannot be reached.
- The API's published description is fixed by a test, so a breaking change shows in review.

### 23. Egress Bundle — Export to Downstream Consumers

**As** an operator handing a run to another system,
**I want** one finished run exported with its original documents as an export package that describes itself,
**so that** other systems can review and reuse the results beside the originals without calling back into Womblex.

**Given** a finished run, kept on a local machine or in cloud storage
**When** the operator runs the export command with a destination
**Then** an export package is written at the destination holding the unchanged results, one copy of each original document, an index of the originals and a description file, and Womblex stops.

**Acceptance criteria:**

- The destination can be any storage address (a local folder, S3, MinIO or GCS), all through the same code.
- `corpus/` copies the run's results folder in full, including every later step's files, plus the run's document list, unchanged.
- Original documents are written to `sources/<source_hash><ext>`, one copy per fingerprint. Two documents with the same fingerprint share one file.
- `source_index.parquet` has one row per document with `source_hash`, `doc_id`, `filename`, `ext`, `raw_key` and a status of `resolved`, `hash_mismatch`, `not_found`, `unsupported_basis` or `upload_failed`. `raw_key` is empty wherever no file was written.
- A problem with one original never stops the export. A missing, mismatched, pre-extracted or failed-upload original is reported in the index and in `egress_manifest.json`.
- `egress_manifest.json` holds the contract version and a report for each document.
- A results-only export (`--no-sources` / `--corpus-only`) does not look for originals and writes no `sources/` or `source_index.parquet`.
- `--source-root` finds the originals for results that have been moved. `--run-id` defaults to the run folder's name and must match the run's own record, if it has one.
- Exporting again into an existing package removes original files the new export neither wrote nor tried to write, and leaves an original whose upload failed untouched.
- A run kept in cloud storage is copied directly from one cloud location to another, with no local copy first.
- For a run whose documents were uploaded to cloud storage, each original is fetched from where it was first uploaded, by its recorded location, and checked against its recorded fingerprint, with the same statuses for each document.
- Womblex writes the export package and nothing beyond it: it does not keep old packages, serve them, or take back corrections made to them. Sending the package to a place a workflow names is Requirement 27.

### 27. Workflows — Set Up Once, Check First, Run Anywhere

**As** an operator,
**I want** to write down a set of processing steps once and run it on my own machine, in the cloud, or on a timetable,
**so that** every run follows the same steps, problems are caught before any money is spent, a timetable only pays for new documents, and the results reach the systems that use them.

**Given** a workflow file that lists the starting settings, any changes to them, where the documents are, the steps to run, when to run them and where to send the results
**When** the operator checks, prices, runs, sends to the cloud, or schedules the workflow
**Then** any problem is reported before the first step starts, each scheduled run handles only new documents, every output shows which workflow made it, and each place named receives the results.

**Acceptance criteria:**

- A workflow is a plain text file. Creating or changing one needs no programming.
- Every command checks the file first and rejects anything it does not recognise.
- Changing only the layout or comments in the file does not change its identity.
- Settings in the workflow are added on top of its starting settings, and the result is checked as a whole.
- The check command runs every test a real run would run before its first step, and changes nothing.
- The pricing command shows how much text each paid step would send, including every step that uses a paid service, and counts text the way each service counts it. It does not use any paid service itself. Steps that run on our own machines are shown as free.
- Steps listed in the wrong order are rejected before anything runs, with a clear reason.
- If a step fails after the run has started, the other steps still run. Each later step notes the failure, and the run is reported as failed.
- Every output file shows which workflow made it. The full workflow is kept with the run, so it can be read even after the file has changed.
- Sending a workflow to the cloud is rejected before anything starts if it would be rejected on a local machine, uses a step the cloud cannot run, or lists steps in an order the cloud would change.
- Scheduling a workflow saves a copy of it and sets its timetable in one action. The saved times and the real timetable always match.
- Scheduling runs every check a scheduled run would make, against the document location named in the workflow. A workflow that would fail is not scheduled.
- Each scheduled run uses the saved copy and reads from the document location named in the workflow, whichever machine runs it.
- A scheduled workflow keeps adding to one run. Each time it runs, it processes only documents the run has not seen before.
- A document already in the run is never processed again or sent to a paid service again.
- After a workflow is changed and scheduled again, the new version applies to new documents only. Earlier documents keep their results.
- If there are no new documents, nothing runs and the log says so.
- A file whose contents have changed counts as a new document. The earlier version stays in the run.
- A workflow has no owner. Only administrators can see the runs it starts, as with runs started from the command line.
- When a workflow includes the step that hides personal details, cloud runs carry it out. Runs started without a workflow never include it.
- A workflow can name places to send its results: cloud file storage, a Postgres database with search vectors, and a web address that receives notices.
- The workflow names where Womblex can find passwords and keys, never the passwords or keys themselves. No secret is ever written in the workflow file or in the copy Womblex keeps.
- Every place is checked before the run starts. Its passwords and keys must be available, and the software it needs must be installed. If a check fails, the run does not start.
- Sending to the database needs the step that creates search vectors. Sending text with personal details hidden needs the step that hides them. A workflow missing the step it needs does not start.
- Results are sent after the last step, even if a step failed. Results sent after a failure say which steps failed.
- If sending fails, Womblex tries again a set number of times, then records the failure on the run. A failed send never stops the run.
- When a failed step is retried successfully, Womblex shows how to resend the results. A resend replaces what was sent before: files are rewritten and database rows are updated, not duplicated.
- Sending to cloud file storage writes the same export package as Requirement 23. Each place decides whether the original documents are included.
- Sending to the database writes one row per piece of text, with its search vector. Text with personal details hidden is sent by default. The original text is sent only where the workflow asks for it, and each row says which version it holds.
- Sending to a web address sends five kinds of notice: all documents read in, a step finished or failed, the run finished, results sent, and sending failed. Each notice is signed with a secret key kept on the machine, so the receiver can trust it.
- Each notice gives the run's status in the same words the web API uses. No notice is sent for each document or group of documents.
- The run's summary record lists every send.

**Out of scope:** removing documents that have changed or been deleted, catching up on scheduled times that were missed, managing workflows through the web API, workflow owners, sending each step's results to a different place, databases other than Postgres with search vectors (each needs its own requirement), setting up notice addresses through the web API, and taking corrections back from the systems that receive the results.

### 28. Cloud Runs — Do What Was Asked, Report What Happened

**As** an operator, or a system that sends work to Womblex,
**I want** work sent to the cloud to use the settings I chose, run only on machines that can handle it, and show its real status after a fix,
**so that** cloud results match local results, and a fixed run is no longer shown as failed.

**Given** a run sent to the cloud from the command line, the web console, the web API or a workflow
**When** the cloud machines process it, a step fails, or the operator retries a failed step
**Then** every part of the run uses the settings that were sent, runs only on a machine checked for it, redoes only the work a failure affected, and shows its real status.

**Acceptance criteria:**

- Every part of a cloud run uses the settings it was sent with. A machine's own settings only say where its files are kept.
- Settings sent through the web API, the web console or the command line control how the run works, not just which steps it includes.
- A cloud machine only runs a step after checking it has the AI models that step needs.
- If no running machine can do part of the work, the run is rejected before anything starts.
- A retry is rejected before it starts for any reason a new run would be rejected.
- When a failed step is retried, later steps redo only the groups of documents the failure affected. Work that already finished correctly is not sent to a paid service again.
- Once every failed step in a run has been retried successfully, the run shows as done.
- A scheduled run whose steps could not be started at first shows its real status once they are started, not "failed".

**Out of scope:** retrying steps automatically, and retrying from the web API or the web console.

## How this plan works
This plan lists only the work still to do. Finished work is recorded in the change log and the other project documents. The rules every piece of work follows, such as how large one change can be and the review at the end of each phase, are in the project guide (CLAUDE.md). Each item is removed from this plan when it is done, and the plan is deleted when the last item is done.

When a piece of work reverses an earlier decision, the old decision is deleted from the decision records. It is not kept as a rejected option. The list of mistakes below is the record of what went wrong.

Each work item names the requirements in the target state it delivers. When a piece of work meets a requirement, the requirement moves into the requirements document.

## Words used in this plan
- **Workflow:** a saved recipe. It names the settings to use, the steps to run, where the documents are, and when to run.
- **Step:** one stage of processing, such as reading the text, finding names, or creating search vectors.
- **Run:** one pass of processing over a set of documents.
- **Group of documents:** the documents processed together as one unit within a run.
- **Cloud machines:** the computers that process runs sent to the cloud.
- **Paid service:** an outside service that charges for each piece of text it processes.
- **Export package:** the folder of results, and optionally original documents, that Womblex prepares for another system.
- **Scheduled run:** a run started automatically at a set time.

## Mistakes to avoid
Phase 2 made each of these mistakes. Check every item in this plan, and every new plan, against this list before building.

1. **Saving a setting before anything uses it.** Workflows got a "send results to" setting that did nothing, which led to questions about settings with no effect. *Rule:* a setting is added in the same piece of work that makes it do something.
2. **Writing a file nobody reads.** Workflow "conditions" wrote a file of labels that nothing reads. The same labels can be worked out from the run's document list at any time. *Rule:* every new output names who reads it. Anything that can be worked out from saved data is worked out when needed, not saved again.
3. **Automatic runs that pay twice.** Each scheduled run processes every document again and pays the outside service again for each one. *Rule:* anything that runs on its own processes only documents it has not seen before.
4. **Taking away the operator's tools.** A workflow could only be created by writing program code. *Rule:* anything an operator sets up has a settings file and a command. An operator never writes code to use a feature.
5. **A reason that was never delivered.** Workflows were stored in a database rather than a file because a file "cannot hold a timetable". The timetable was then built so that it repeats all the work every time (mistake 3). *Rule:* when a feature is the reason for a decision, that feature must work properly in the same phase, or the decision is reopened.
6. **Stopping a run after it started, then undoing it.** Two changes stopped work part way through a run because of a single document. Both broke the rule that a run, once started, always finishes, and both had to be undone. *Rule:* every check that can stop a run happens before the run starts.
7. **Passwords in saved settings.** A planned setting would have saved a database password inside the stored workflow. *Rule:* saved settings name where a password is kept, never the password itself.
8. **Asking the owner questions the design already answers.** Reviews treated unfinished pieces as settled and asked the owner to choose between options. *Rule:* only ask the owner a question if the answer changes what users get. Anything the written design already answers is settled by the builder, who says where.

## Phase R — Repairs

### Decisions made on 17 October 2026
Each piece of work records its decision in the operating decisions document and deletes any decision it reverses.

- **Decision 1. Workflows are files.** A workflow is a plain text settings file, and every workflow command takes the file's location. The file is the master copy. Womblex keeps its own copy only when a workflow is scheduled, because scheduled runs read that copy. Sending a workflow to the cloud keeps no copy, because the cloud run receives the whole workflow when it starts. This reverses the decision that workflows are created only through code.
- **Decision 2. A scheduled workflow adds to one run instead of starting again.** Each scheduled run checks every file in the document folder, processes only the documents the run does not already have, and runs the steps on those new documents only. A file whose contents change counts as a new document, and the old version stays. Removing old versions is not part of this work.
  Each scheduled run uses the workflow as it is when that run starts. After an edit, the new version applies to new documents only. The rule that refused a run already used by another version of a workflow is removed, because it no longer serves a purpose.
- **Decision 3. Conditions are removed.** Everything they produced can be found from the run's document list.
- **Decision 4. The "send results to" setting is removed.** It returns in Phase 3, once results can actually be sent.

### Decisions proposed on 17 October 2026, not yet made
These fix problems found in a review of the work done between 10 and 17 October.

- **Decision 5. A run's status reflects the latest try at each piece of work.** Today, a run whose failed step was retried successfully still shows as failed, and so does a scheduled run that was fixed by hand. After this change, both show their real status. This reverses the earlier rule that the original failure keeps counting.
- **Decision 6. A retry redoes only the work the failure affected.** Today, retrying a step that ran after a failure redoes every group of documents and pays the outside service again for all of them. After this change, it redoes only the groups the failure affected.
- **Decision 7. Every cloud run uses the settings it was sent with.** Today, only workflows do. Runs sent from the web API, the web console or the command line use each cloud machine's own settings, so the same run can behave differently on different machines. After this change, every run carries its settings with it. This reverses the part of Requirement 22 that says submitted settings only choose which steps run.
- **Decision 8. Work goes only to machines checked for it.** Today, a machine checks only the AI models its own settings use, but accepts work that needs other models. After this change:
  - each piece of work goes only to machines that have checked the models it needs;
  - a machine checks the models for every step in its settings, whether or not that step is turned on;
  - each machine records which kinds of work it accepts;
  - if no running machine can do the work, the work is refused before anything starts;
  - a retry runs every check a new run would.
- **Decision 9. A workflow names where its documents are stored.** Today, a scheduled run reads from whichever storage the machine that picks it up uses. After this change, the workflow names the storage, and only machines that use it run the workflow. Passwords for the storage still come from the machine, never the workflow.
- **Decision 10. Scheduling saves the workflow and its timetable together.** Today, a stored workflow can be changed without updating its timetable, so the saved times and the real timetable can differ. After this change, only scheduling saves a workflow, and it runs the checks and updates the timetable in the same action.
- **Decision 11. A run records the whole workflow that made it.** Today, a run records only a code for the workflow's version, and once the workflow changes the original cannot be found. After this change, each output file and the run's summary record hold the workflow's name and full contents.
- **Decision 12. The pricing command counts every paid call.** Today, it misses the paid calls made when text is split into pieces using AI, and it counts text for every outside service as if it were the default one. After this change:
  - AI text-splitting calls are counted;
  - text is counted the way each service counts it;
  - steps that run on Womblex's own machines are shown as free.
- **Decision 13. Workflows have no owner.** No command or screen lets anyone set an owner, so the owner setting does nothing, and it is removed. Runs started by a workflow are visible to administrators, as runs started from the command line are. An owner can be added back if the web API ever lets users create workflows. This reverses the rule that workflows are owned like runs.

### Work items
Each item is one change that stays under the project's size limit. An item that grows too large is split into two changes that each work on their own.

- **R.1 Remove conditions** (Decision 3; Requirement 27) from both local and cloud runs, with their documentation.
- **R.2 Remove the "send results to" setting** (Decision 4; Requirement 27) and its open item in the requirements.
- **R.2b Remove the workflow owner** (Decision 13; Requirements 22 and 27), including the owner check on scheduled runs.
- **R.3 Workflows as files** (Decisions 1 and 10; Requirement 27). Every workflow command takes a file, and an example workflow is added beside the example settings. Only scheduling saves a copy, and it updates the timetable in the same action. The code example in the workflow guide is removed.
- **R.3b A workflow names its document storage** (Decision 9; Requirement 27). Scheduled runs go only to machines that use that storage.
- **R.4 Scheduled runs add only new documents** (Decision 2; Requirement 27), and the rule against a run used by two versions of a workflow is removed. If too large, the change to internal labels goes first and the new-documents-only behaviour second.
- **R.5 Run status from the latest try** (Decision 5; Requirement 28).
- **R.6 Every cloud run carries its settings** (Decision 7; Requirements 22 and 28), with the web API guide updated.
- **R.7 Work goes only to checked machines** (Decision 8; Requirements 14 and 28). If too large, the way work is sorted goes first and the up-front refusal second.
- **R.8 Retries redo only affected work** (Decision 6; Requirements 14 and 28).
- **R.9 Runs record their whole workflow** (Decision 11; Requirement 27). The run's summary record gets a new version number.
- **R.10 Pricing counts every paid call** (Decision 12; Requirement 27).
- **R.11 End-of-phase review.**

## Phase 3 — Sending results to other systems
In Phase 3, a finished run sends its results to other systems by itself, to the places its workflow file names. Each piece of work adds one kind of place that works from start to finish. No kind of place can be named before Womblex can deliver to it.

### Decisions made on 17 October 2026
Each is recorded with the piece of work that builds it.

- **Cloud runs hide personal details when a workflow asks.** Runs started without a workflow still never include this step. This reverses the rule that cloud runs never include it.
- **Cloud runs write their own document list when they finish.** The command that writes it by hand stays, and running it again changes nothing.
- **A run kept in cloud storage is exported directly between cloud locations.** It no longer has to be copied to a local machine first.
- **Status uses the web API's existing words** (pending, running, done, failed and cancelled, plus a warning for a scheduled run that could not start its steps). The status sentence, the notices and the run's history record all use them. This replaces a different set of words chosen earlier.
- **Sending results follows the retry rules.** Results sent after a failure are marked with that failure. When the failed step is retried successfully, the retry command shows how to resend the results. Nothing is resent automatically.
- **Places to send results are checked before the run starts.** Passwords, the signing key for notices and any extra software needed are checked alongside the AI model check.
- **Sending to the database needs search vectors.** A workflow without the step that creates them is refused before it starts.
- **The requirements for sending results belong in Requirement 27.**

### Work items
- **3.1 New software only.** The library for sending notices becomes part of the standard install, and the database's library becomes an optional add-on. This needs human approval.
- **3.2a Cloud runs write their own document list** (Requirement 2). The command and the cloud use the same code.
- **3.2b Export from cloud storage** (Requirement 23). A run kept in cloud storage is exported directly. Original documents are fetched from where they were first uploaded and checked against their recorded fingerprint.
- **3.2c Sending to cloud file storage, and how sending works** (Requirement 27). The "send results to" setting returns, with cloud file storage as its only kind. Results are sent after the last step, even if a step failed, and a failed send is retried a set number of times. The run's summary record lists every send. A failed send is recorded on the run and never stops it. The up-front checks and the retry instructions are added here.
- **3.3 Hiding personal details in cloud runs** (Requirement 27).
- **3.4 Sending to a Postgres database with search vectors** (Requirement 27). The workflow names where the database password is kept. Each piece of text is one row, updated rather than duplicated when results are resent. Text with personal details hidden is sent by default, and original text only where the workflow asks, with each row saying which version it holds. A workflow that sends hidden text without the hiding step, or sends to the database without the search-vector step, is refused before it starts.
- **3.5 Notices to a web address** (Requirements 22 and 27). Each notice is retried a set number of times and signed with a secret key kept on the machine.
- **3.5.5 The export guide also covers sending results**, with a description of what is sent. Documentation only.
- **3.6 End-of-phase review.**

## Phase 4 — Day-to-day operation

### Work items
- **4.0 Split one large command-line file.** The command that lists jobs moves to its own file, because the current file is near the size limit. Commands and output do not change.
- **4.1 One-sentence run status** (Requirements 16 and 22). One plain sentence for each run, built from the exact counts at each step, shown by the command line, the web API and the web console. The run's history record reports finished, failed or cancelled, instead of always finished.
- **4.2 A separate file for re-linked names** (Requirement 12). The step that re-links names to text writes its own file, marked with the exact version of the text it linked to, and readers use it where it matches. Until 4.3, the step also keeps rewriting the old files.
- **4.3 Re-linking stops changing earlier results** (Requirement 12). Older runs are read as before. The page-layout step keeps replacing its own file.
- **4.3.5 Documentation for 4.2 and 4.3.**
- **4.4 Final review.** The list of mistakes moves to the dead-ends document. The rest of this plan moves into the decision records and the change log, and the plan is deleted.
