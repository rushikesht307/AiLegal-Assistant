"""
Intent Router 

Solves Use Case 3 properly: when a document is uploaded, decide whether the
user's question is ABOUT THE DOCUMENT or a GENERAL legal question.

Two-layer, flexible approach (no fragile keyword lists):

  Layer 1 - LLM intent classification (primary):
            ask Gemini to label the question as "document" or "general".
            Robust because it understands meaning, not keywords.

  Layer 2 - Retrieval-confidence fallback (safety net):
            if the LLM is unsure/unavailable, retrieve from the document and
            check whether the retrieved context is actually relevant.
            If not relevant -> treat as general -> CUAD.
"""


class IntentRouter:

    def __init__(self, generator):
        """
        generator : the Gemini Generator (has a .generate(prompt) method)
        """
        self.generator = generator

    # ---------------- Layer 1: LLM intent classification ----------------
    def classify_intent(self, question: str) -> str:
        """
        Ask the LLM whether the question is about the uploaded document
        or a general legal question.
        Returns "document" or "general" (or "unknown" if the LLM fails).
        """
        prompt = f"""You are an intent classifier for a legal assistant.
A user has uploaded a specific legal document and is now asking a question.
Decide if the question is:
  - "document": asking about the content of THEIR uploaded document
                (e.g., "what is the notice period in this", "who are the parties",
                 "summarize this contract", "what does clause 3 say").
  - "general" : asking a general legal question NOT specific to their document
                (e.g., "what is an NDA", "explain confidentiality clauses",
                 "what is a typical lease term").

Question: "{question}"

Reply with ONLY one word: document OR general."""

        try:
            result = self.generator.generate(prompt).strip().lower()
            if "general" in result:
                return "general"
            return "document"
        except Exception:
            # LLM unavailable -> let the retrieval layer decide
            return "unknown"

    # ---------------- Layer 2: retrieval-confidence check ----------------
    def is_document_relevant(self, question: str, document_retriever, min_len: int = 40) -> bool:
        """
        Retrieve from the document and check if the context is actually useful.
        Returns True if the document seems to contain a relevant answer.
        """
        if document_retriever is None:
            return False
        try:
            context = document_retriever.search_documents(question)
        except Exception:
            return False
        # if the document returned very little text, it's probably not relevant
        return bool(context and len(context.strip()) >= min_len)

    # ---------------- Combined decision ----------------
    def route(self, question: str, has_document: bool, document_retriever=None,
              general_hint: bool = False) -> str:
        """
        Decide the final source for a question.
        Returns "document" or "knowledge".

          - no document uploaded              -> "knowledge" (CUAD)
          - explicit general_hint from the UI -> "knowledge" (CUAD)
          - document uploaded:
                Layer 1 (LLM) says general     -> "knowledge"
                Layer 1 (LLM) says document    -> "document"
                Layer 1 unknown -> Layer 2:
                    document relevant          -> "document"
                    not relevant               -> "knowledge"
        """
        # Use case 1: no document -> always CUAD
        if not has_document:
            return "knowledge"

        # explicit hint from the UI (optional shortcut)
        if general_hint:
            return "knowledge"

        # Use case 2 vs 3: classify intent with the LLM
        intent = self.classify_intent(question)

        if intent == "general":
            return "knowledge"        # Use case 3 -> CUAD
        if intent == "document":
            return "document"         # Use case 2 -> uploaded document

        # LLM unsure -> fall back to retrieval-confidence
        if self.is_document_relevant(question, document_retriever):
            return "document"
        return "knowledge"