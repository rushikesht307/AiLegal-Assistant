from agents.legal_rag_agent.qa_agent import QAAgent
from agents.clause_extraction_agent.clause_extractor import ClauseExtractionAgent
from agents.risk_analysis_agent.risk_agent import RiskAnalysisAgent
from agents.compliance_agent.compliance_agent import ComplianceAgent
from agents.contract_comparison_agent.comparison_agent import ContractComparisonAgent
# from agents.obligation_agent.obligation_agent import ObligationAgent
# from agents.report_generator_agent.report_generator import ReportGeneratorAgent


class Planner:

    def __init__(self, pipeline):
        """
        pipeline : the RAGPipeline instance (gives access to retrievers, generator, memory).
        """
        self.pipeline = pipeline

    def _build(self, agent_cls):
        """Helper: build an agent with the current retrievers + generator + memory."""
        # make sure the retrievers exist
        self.pipeline._get_knowledge_retriever()
        return agent_cls(
            knowledge_retriever=self.pipeline.knowledge_retriever,
            document_retriever=self.pipeline.document_retriever,
            generator=self.pipeline.generator,
            memory=self.pipeline.memory,
            router=self.pipeline.router,
        )

    def route(self, question, has_document=False, general=False):
        """
        Read the question, pick the right agent, run it.
        Returns: { "agent": <name>, "answer": <text>, "mode": <document|knowledge> }
        """
        q = question.lower()

        # make sure the document retriever exists if a doc is uploaded
        if has_document:
            self.pipeline._get_document_retriever()

        # ---- Report generation (ONLY when asked) ----
        # if "report" in q or "generate report" in q or "full analysis" in q:
        #     agent = self._build(ReportGeneratorAgent)
        #     return agent.run(question, has_document, general)

        # ---- Risk ----
        if "risk" in q or "risky" in q or "red flag" in q or "danger" in q:
            agent = self._build(RiskAnalysisAgent)
            return agent.run(question, has_document, general)

        # ---- Compliance ----
        if "compliance" in q or "compliant" in q or "regulation" in q or "gdpr" in q:
            agent = self._build(ComplianceAgent)
            return agent.run(question, has_document, general)

        # ---- Contract comparison ----
        if "compare" in q or "difference" in q or "vs " in q or "version" in q:
            agent = self._build(ContractComparisonAgent)
            return agent.run(question, has_document, general)

        # ---- Obligations / deadlines ----
        # if "deadline" in q or "obligation" in q or "renew" in q or "due date" in q or "notice period" in q:
        #     agent = self._build(ObligationAgent)
        #     return agent.run(question, has_document, general)

        # ---- Clause extraction ----
        if "clause" in q or "extract" in q or "termination" in q or "confidential" in q or "liability" in q:
            agent = self._build(ClauseExtractionAgent)
            return agent.run(question, has_document, general)

        # ---- Default: Legal Q&A ----
        agent = self._build(QAAgent)
        result = agent.answer(question, has_document, general)
        return {"agent": "Legal Q&A Agent", **result}
