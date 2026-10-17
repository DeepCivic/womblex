# Trust and Recipes — Plan

*Status: in progress, October 2026.*

Phase 1 is finished. Phase 2 is finished but has design mistakes, and Phase R fixes them before Phase 3 starts.

This plan lists only the work still to do. Finished work is recorded in the change log and the other project documents. The rules every piece of work follows, such as how large one change can be and the review at the end of each phase, are in the project guide (CLAUDE.md). Each item is removed from this plan when it is done, and the plan is deleted when the last item is done.

When a piece of work reverses an earlier decision, the old decision is deleted from the decision records. It is not kept as a rejected option. The list of mistakes below is the record of what went wrong.

Each phase ends with draft requirements, written in the same form as the requirements document. When a piece of work meets a requirement, the requirement moves into that document.

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

- **R.1 Remove conditions** (Decision 3) from both local and cloud runs, with their documentation.
- **R.2 Remove the "send results to" setting** (Decision 4) and its open item in the requirements.
- **R.2b Remove the workflow owner** (Decision 13), including the owner check on scheduled runs.
- **R.3 Workflows as files** (Decisions 1 and 10). Every workflow command takes a file, and an example workflow is added beside the example settings. Only scheduling saves a copy, and it updates the timetable in the same action. The code example in the workflow guide is removed.
- **R.3b A workflow names its document storage** (Decision 9). Scheduled runs go only to machines that use that storage.
- **R.4 Scheduled runs add only new documents** (Decision 2), and the rule against a run used by two versions of a workflow is removed. If too large, the change to internal labels goes first and the new-documents-only behaviour second.
- **R.5 Run status from the latest try** (Decision 5).
- **R.6 Every cloud run carries its settings** (Decision 7), with the web API guide updated.
- **R.7 Work goes only to checked machines** (Decision 8). If too large, the way work is sorted goes first and the up-front refusal second.
- **R.8 Retries redo only affected work** (Decision 6).
- **R.9 Runs record their whole workflow** (Decision 11). The run's summary record gets a new version number.
- **R.10 Pricing counts every paid call** (Decision 12).
- **R.11 End-of-phase review.**

### Draft requirements — Phase R
**Requirement 27 is replaced by:**

#### 27. Workflows — Set Up Once, Check First, Run Anywhere

**As** an operator,
**I want** to write down a set of processing steps once and run it on my own machine, in the cloud, or on a timetable,
**so that** every run follows the same steps, problems are caught before any money is spent, and a timetable only pays for new documents.

**Given** a workflow file that lists the starting settings, any changes to them, where the documents are, the steps to run, and when to run them
**When** the operator checks, prices, runs, sends to the cloud, or schedules the workflow
**Then** any problem is reported before the first step starts, each scheduled run handles only new documents, and every output shows which workflow made it.

**Acceptance criteria:**

- A workflow is a plain text file. Creating or changing one needs no programming.
- Every command checks the file first and rejects anything it does not recognise.
- Changing only the layout or comments in the file does not change its identity.
- Settings in the workflow are added on top of its starting settings, and the result is checked as a whole.
- The check command runs every test a real run would run before its first step, and changes nothing.
- The pricing command shows how much text each paid step would send, including every step that uses a paid service. It does not use any paid service itself. Steps that run on our own machines are shown as free.
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

**Out of scope:** removing documents that have changed or been deleted, catching up on scheduled times that were missed, managing workflows through the web API, workflow owners, and sending results to other systems (Phase 3).

**New requirement:**

#### 28. Cloud Runs — Do What Was Asked, Report What Happened

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

This replaces the part of Requirement 22 that says submitted settings only choose which steps run, and adds to the retry part of Requirement 14.

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
- **3.2a Cloud runs write their own document list.** The command and the cloud use the same code.
- **3.2b Export from cloud storage.** A run kept in cloud storage is exported directly. Original documents are fetched from where they were first uploaded and checked against their recorded fingerprint.
- **3.2c Sending to cloud file storage, and how sending works.** The "send results to" setting returns, with cloud file storage as its only kind. Results are sent after the last step, even if a step failed, and a failed send is retried a set number of times. The run's summary record lists every send. A failed send is recorded on the run and never stops it. The up-front checks and the retry instructions are added here.
- **3.3 Hiding personal details in cloud runs.**
- **3.4 Sending to a Postgres database with search vectors.** The workflow names where the database password is kept. Each piece of text is one row, updated rather than duplicated when results are resent. Text with personal details hidden is sent by default, and original text only where the workflow asks, with each row saying which version it holds. A workflow that sends hidden text without the hiding step, or sends to the database without the search-vector step, is refused before it starts.
- **3.5 Notices to a web address.** Each notice is retried a set number of times and signed with a secret key kept on the machine.
- **3.5.5 The export guide also covers sending results**, with a description of what is sent. Documentation only.
- **3.6 End-of-phase review.**

### Draft requirements — Phase 3
**Requirement 27 adds these, as each kind of destination is built (3.2c, 3.3, 3.4 and 3.5):**

The workflow file can also name places to send the results. Once the steps have run, each place named receives the results.

- A workflow can name places to send its results. Womblex accepts a kind of place only once it can actually deliver to it: cloud file storage first, then a Postgres database with search vectors, then a web address that receives notices.
- The workflow names where Womblex can find passwords and keys, never the passwords or keys themselves. No secret is ever written in the workflow file or in the copy Womblex keeps.
- Every place is checked before the run starts. Its passwords and keys must be available, and the software it needs must be installed. If a check fails, the run does not start.
- Sending to the database needs a step that creates search vectors. Sending text with personal details hidden needs the step that hides them. A workflow missing the step it needs does not start.
- Results are sent after the last step, even if a step failed. Results sent after a failure say which steps failed.
- If sending fails, Womblex tries again a set number of times, then records the failure on the run. A failed send never stops the run.
- When a failed step is retried successfully, Womblex shows how to resend the results. A resend replaces what was sent before. Files are rewritten and database rows are updated, not duplicated.
- Sending to cloud file storage writes the same export package as Requirement 23. Each place decides whether the original documents are included.
- Sending to the database writes one row per piece of text, with its search vector. Text with personal details hidden is sent by default. The original text is sent only where the workflow asks for it, and each row says which version it holds.
- Sending to a web address sends five kinds of notice: all documents read in, a step finished or failed, the run finished, results sent, and sending failed. Each notice is signed with a secret key so the receiver can trust it.
- Each notice gives the run's status in the same words the web API uses. No notice is sent for each document or group of documents.
- When a workflow includes the step that hides personal details, cloud runs carry it out. Runs started without a workflow still never include it.

**Requirement 23 changes (3.2b):** Export works for a finished run whether it is kept on a local machine or in cloud storage. A run kept in cloud storage is copied directly from one cloud location to another, and its original documents are fetched from where they were first uploaded. The TO-DO and its Outstanding entry are removed.

**Requirement 2 adds (3.2a):** When a cloud run's steps finish, Womblex writes the run's list of documents by itself. The command that writes the list by hand still works, and running it again changes nothing.

**Requirement 22 adds (3.5):** A run started from a workflow that names a web address sends its notices to that address. Callers can still check a run's status themselves, and a caller with no web address does so.

**Out of scope for Phase 3:** sending each step's results to a different place, databases other than Postgres with search vectors (each needs its own requirement), setting up notice addresses through the web API, and taking corrections back from the systems that receive the results.

## Phase 4 — Day-to-day operation

### Work items
- **4.0 Split one large command-line file.** The command that lists jobs moves to its own file, because the current file is near the size limit. Commands and output do not change.
- **4.1 One-sentence run status.** One plain sentence for each run, built from the exact counts at each step, shown by the command line, the web API and the web console. The run's history record reports finished, failed or cancelled, instead of always finished.
- **4.2 A separate file for re-linked names.** The step that re-links names to text writes its own file, marked with the exact version of the text it linked to, and readers use it where it matches. Until 4.3, the step also keeps rewriting the old files.
- **4.3 Re-linking stops changing earlier results.** Older runs are read as before. The page-layout step keeps replacing its own file.
- **4.3.5 Documentation for 4.2 and 4.3.**
- **4.4 Final review.** The list of mistakes moves to the dead-ends document. The rest of this plan moves into the decision records and the change log, and the plan is deleted.

### Draft requirements — Phase 4
**Requirements 16 and 22 add (4.1):**

- A run's status is shown as one plain sentence, built from the exact counts of work done, waiting and failed at each step. The counts are still shown beside it.
- The sentence uses the same status words as the web API (pending, running, done, failed and cancelled), plus a warning when a scheduled run could not start its steps.
- The command line, the web API and the dashboard all show the same sentence for the same run.
- The run's history record, which other tools use to track where data came from, reports the run's real outcome: finished, failed or cancelled. Today it always says finished.

**Requirement 12 changes (4.2 and 4.3):** the step that re-links named people and organisations to the pieces of text they appear in no longer rewrites earlier results. Instead:

- It saves its links in a file of their own, marked with the exact version of the text pieces it linked to. It never changes the files the earlier step wrote.
- Anyone reading the results can tell when the links no longer match the current text pieces, and the matching links are used wherever they exist.
- Runs made before this change are read exactly as before.
