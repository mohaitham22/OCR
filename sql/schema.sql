-- OCR DMS schema: extracted documents, and every correction a reviewer made.
--
-- Applied by app.db.init_schema(), which runs this file whole. Every statement
-- is IF NOT EXISTS, so running it against a live database is a no-op rather
-- than an error -- that is what lets a deployment apply it on boot without
-- anyone deciding first whether it has been applied before.

-- Trigram search on raw_text. See the index at the bottom of this file for why
-- this and not a tsvector.
CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- ---------------------------------------------------------------------------
-- documents
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS documents (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Provenance. `source` is the uploaded filename; `content_hash` is a
    -- digest of the file bytes, so the same page uploaded twice is findable as
    -- a duplicate instead of being extracted and paid for twice.
    source          text,
    content_hash    text,
    doc_type        text        NOT NULL,

    -- Which backend produced this, for audit and for compare mode. Nothing
    -- outside app/engines/ branches on it; it is recorded, not consulted.
    engine          text,
    model           text,

    -- The review gate. app.validate.decide_status chooses between the first
    -- two on insert, and the default is the conservative one.
    status          text        NOT NULL DEFAULT 'pending_review'
                    CHECK (status IN ('pending_review', 'approved', 'rejected')),

    -- The two payloads, and the reason there are two of them.
    --
    -- `fields` is what the business queries: the current, best-known values.
    -- A reviewer's approval REPLACES it, so it always holds the corrected
    -- truth and a report built on it needs no knowledge of review history.
    --
    -- `raw_extraction` is the exact output of the engine that read the page --
    -- the whole ExtractionResult, including the document as the model first
    -- returned it, the OCR blocks and their confidences. It is written once,
    -- on insert, and NEVER updated: not by an approval, not by a correction,
    -- not by a migration.
    --
    -- Keep it forever. This schema will change -- a field we do not extract
    -- today becomes a column next quarter; a field stored as text turns out to
    -- need a currency beside it; a bug in the parser turns out to have been
    -- dropping the tax line all along. When that happens the choice is either
    -- to re-derive the new shape from raw_extraction for every document
    -- processed months ago, in one UPDATE that costs nothing, or to send the
    -- whole archive back through inference and pay for it again -- assuming
    -- the original files still exist, which for a system fed by uploads is not
    -- a safe assumption. Storage is cheap; re-inference is not. And the
    -- decision is impossible to backfill: raw output that was never written
    -- down cannot be recovered later, at any price.
    fields          jsonb       NOT NULL DEFAULT '{}'::jsonb,
    raw_extraction  jsonb,

    -- Full transcribed page text, kept beside the structured fields so a
    -- reviewer can check a value against what was on the paper, and so a
    -- document nobody has structured well is still findable by search.
    raw_text        text,

    -- What app.validate found, as returned: [{code, message, severity, field,
    -- expected, actual}]. Stored rather than recomputed, because the rules
    -- change and this is what the reviewer was actually shown.
    issues          jsonb       NOT NULL DEFAULT '[]'::jsonb,

    -- Overall recogniser confidence, 0.0-1.0. NULL is *no score*, not a low
    -- one: the vision engines and the PDF text-layer path recognise nothing
    -- and report NULL by design, which is why the CHECK admits it.
    confidence      real        CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    page_count      integer,
    duration_ms     real,

    -- Compare mode: the second engine's ExtractionResult, and the fraction of
    -- fields on which the two agreed. Both NULL when only one engine ran.
    -- Stored for the same reason raw_extraction is -- two engines disagreeing
    -- on a page is the cheapest signal this system produces about which one to
    -- trust, and it is only produced once, at the moment both were run.
    comparison      jsonb,
    agreement       real        CHECK (agreement IS NULL OR (agreement >= 0 AND agreement <= 1)),

    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    reviewed_at     timestamptz,
    reviewed_by     text
);


-- ---------------------------------------------------------------------------
-- corrections
-- ---------------------------------------------------------------------------

-- One row per field a reviewer changed: the dotted path, what the engine said,
-- what the human said instead.
--
-- This is not an audit log kept out of politeness. It is the only labelled
-- data this system will ever produce, and it is produced for free by work
-- somebody is doing anyway. Two things get built out of it:
--
--   * The eval set. eval.py measures field-level accuracy, and for that it
--     needs pages whose right answer is known. Every reviewed document is
--     exactly that -- a page, an engine's answer, and a human's answer for the
--     fields where the two differed.
--   * The fine-tuning set, later. Same rows, same shape: the page in, the
--     corrected fields out.
--
-- It cannot be reconstructed after the fact. The moment documents.fields is
-- overwritten with the corrected values, the fact that a human looked at that
-- particular field and disagreed exists nowhere else -- raw_extraction still
-- holds the old answer, but nothing in it says which fields were checked, or
-- which were checked and found right. A reviewer's keystroke is a label, and a
-- label not written down at the instant it is made is gone.
CREATE TABLE IF NOT EXISTS corrections (
    id            bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- RESTRICT, not CASCADE. A correction is only interpretable next to the
    -- page it came from, so deleting a document really does destroy the pair
    -- -- which is the argument for making that deletion something a person has
    -- to write a second statement for, rather than a silent side effect of
    -- tidying up the documents table.
    document_id   uuid        NOT NULL REFERENCES documents (id) ON DELETE RESTRICT,

    -- Dotted path into the document, in the same spelling app.validate uses
    -- for Issue.field: `total`, `items.2.unit_price`. One row per leaf, so
    -- "which field does this engine get wrong most often" is a GROUP BY.
    field_path    text        NOT NULL,

    -- jsonb, not text, so 12.5 and "12.50" stay distinguishable: a training
    -- pair that cannot tell a number from its printed spelling is not much of
    -- a training pair. NULL means the field carried no value -- app.db writes
    -- the same NULL whether the key was absent, explicitly null, or an empty
    -- box in the review form, because for a schema where every extracted field
    -- is optional those are one statement: the document does not show it.
    old_value     jsonb,
    new_value     jsonb,
    CONSTRAINT corrections_actually_changed CHECK (old_value IS DISTINCT FROM new_value),

    -- Copied from the document rather than joined to it, on purpose. A
    -- document can be re-processed by a different engine, and after that
    -- documents.engine names the backend behind the *current* fields, not the
    -- one whose answer this reviewer rejected. Per-engine accuracy is the
    -- first question anyone asks of this table, and a join would answer it
    -- wrongly and silently.
    doc_type      text        NOT NULL,
    engine        text,
    model         text,

    corrected_by  text,
    created_at    timestamptz NOT NULL DEFAULT now()
);


-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- The review queue: "pending_review, newest first". status leads because that
-- is the equality; created_at DESC is in the index because that is the order
-- the queue is read in, and having it there means no sort.
CREATE INDEX IF NOT EXISTS documents_status_created_idx
    ON documents (status, created_at DESC);

-- The same shape for "every invoice, newest first" -- the other listing the UI
-- offers, and the one a monthly report is built from.
CREATE INDEX IF NOT EXISTS documents_doc_type_created_idx
    ON documents (doc_type, created_at DESC);

-- Duplicate detection on upload: has this exact file been through here before.
-- Not UNIQUE -- the same page legitimately arrives twice, re-scanned or
-- re-filed against a different reference, and refusing the insert would lose
-- the second filing rather than flag it.
CREATE INDEX IF NOT EXISTS documents_content_hash_idx
    ON documents (content_hash);

-- Containment queries against the extracted fields:
--   WHERE fields @> '{"merchant_name": "Al Nakheel Market"}'
-- jsonb_path_ops rather than the default jsonb_ops: it indexes one hash per
-- path-and-value instead of an entry per key and per value, which makes it
-- several times smaller and faster on @>. What it gives up is the
-- key-existence operators (? ?| ?&), which this table does not need: every
-- extracted field is optional and therefore present-as-null, so "does this
-- document have a merchant_name" is a question about the value, not the key.
CREATE INDEX IF NOT EXISTS documents_fields_idx
    ON documents USING gin (fields jsonb_path_ops);

-- Free-text search over the transcription, by trigram and not by tsvector.
--
-- The corpus is Arabic, English and mixed, and Postgres ships no Arabic text
-- search configuration: no stemmer, no stop words, no lexeme normalisation.
-- to_tsvector('simple', raw_text) would index whitespace-delimited tokens
-- verbatim, and in Arabic the definite article, the conjunctions and the
-- attached pronouns are written joined to the word, so the same noun appears
-- as several different tokens and a search for one of them matches none of the
-- others. Fixing that properly needs a stemmer Postgres does not have.
--
-- Trigrams do not know what language they are looking at, which is exactly the
-- point here. They also survive OCR: a recogniser that misreads one character
-- in a word leaves most of that word's trigrams intact, so the row still
-- matches, where token equality would have missed it outright. On a corpus
-- defined by being imperfectly transcribed, that is worth more than stemming.
--
-- One requirement comes with them, and it fails silently. pg_trgm decides what
-- a letter is with the *database's* LC_CTYPE, and under LC_CTYPE=C nothing
-- outside ASCII is a letter, so every Arabic character is discarded before the
-- trigrams are cut. Measured on PostgreSQL 16.2, one Arabic word:
--
--   LC_CTYPE=C       show_trgm -> 0 trigrams,  word_similarity 0.0
--   LC_CTYPE=en-US   show_trgm -> 7 trigrams,  word_similarity 1.0
--
-- A C-locale database therefore builds this index, keeps it up to date, and
-- matches no Arabic in it ever -- no error, no empty-index warning, just a
-- query that returns nothing and a corpus that looks like it contains no
-- Arabic. The cluster must be initialised with a UTF-8-aware LC_CTYPE (any
-- will do; the collation itself does not matter here, only the character
-- classification). app.db.init_schema logs a warning when it finds otherwise,
-- because this is not the kind of thing anyone discovers by reading a plan.
--
-- What this buys is fuzzy substring matching, and only that. No relevance
-- ranking, no phrase search, no field weighting, and the index grows with the
-- length of the text rather than with the size of the vocabulary. So this
-- index is the line: if Arabic search quality ever has to be a feature rather
-- than a convenience -- ranked results, morphological matching, snippets --
-- the answer is OpenSearch with an Arabic analyser alongside Postgres, not a
-- cleverer index here. Postgres cannot be taken there, and pretending it can
-- costs a migration to find out.
CREATE INDEX IF NOT EXISTS documents_raw_text_trgm_idx
    ON documents USING gin (raw_text gin_trgm_ops);

-- Every correction for one document: what the review UI shows beside the
-- fields, and the lookup the foreign key itself needs on delete.
CREATE INDEX IF NOT EXISTS corrections_document_idx
    ON corrections (document_id);

-- The eval query: which fields this engine gets wrong, and how often.
CREATE INDEX IF NOT EXISTS corrections_engine_field_idx
    ON corrections (engine, field_path);
