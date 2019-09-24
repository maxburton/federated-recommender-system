## Data
Keep your data (e.g. from evaluations here)


* If you involved human subjects in any form, you will require ethical permission.
    * Keep records of all items related to ethics in `data/ethics`. There are templates for scripts, guidance provided.
    * **You must have scanned PDFs of signed checklists in this folder**, or PDFs of ethics confirmations from other sources
    * Ensure you remain GDPR compliant. In general:
        * Never collect personally identifiable information if at all possible. 
        * Pseudonymise identifiers for subjects. 
        * Use coarse demographic values unless you need specific information (for example, if you need age ranges, collect ranges, not specific ages)
        * Ensure you have explicit consent for the storage and use of data from human subjects
        * DO NOT STORE PERSONALLY IDENTIFIABLE INFORMATION ON REMOTE SERVERS (no Dropbox, Github, etc.)

* Keep a written description of the data, what is contained, and how it was captured in `data/readme.md`
* Record all raw data as an immutable store. **Never modify captured data.** 
    * Keep this under `data/raw`
    * This could be logs, questionnaire responses, computation results

* Write scripts to produced processed data from these (e.g. tidy dataframes, excel sheets, csv files, HDF5 files, sqlite databases)
* Write scripts that process these into results, visualisations, tables that you include in your project.
* If you use Jupyter/RStudio notebooks, place these in `data/notebooks` and name them carefully (not "Untitled1", "Untitled2").

* You may need to remove the `data/` folder from version control if the data size is too large or you are bound by confidentiality.
* If you do so **make sure you have good backups**

