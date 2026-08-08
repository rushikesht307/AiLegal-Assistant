"""
Report Generator Agent   (Owner: Cheruku Alankrutha Reddy)   [Day 5]
Runs the analysis agents, combines their outputs into a full report,
and exports it as a PDF using the export layer.
"""

import os
from agents.clause_extraction_agent.clause_extractor import ClauseExtractionAgent
from agents.risk_analysis_agent.risk_agent import RiskAnalysisAgent
from agents.compliance_agent.compliance_agent import ComplianceAgent
from agents.obligation_agent.obligation_agent import ObligationAgent
from agents.report_generator_agent.pdf_export import export_report_pdf

REPORT_DIR = os.path.join("storage", "generated_reports")
os.makedirs(REPORT_DIR, exist_ok=True)


class ReportGeneratorAgent:

    def __init__(self, knowledge_retriever, document_retriever, generator, memory, router):
        self.knowledge_retriever = knowledge_retriever
        self.document_retriever = document_retriever
        self.generator = generator
        self.memory = memory
        self.router = router

    def _mk(self, agent_cls):
        return agent_cls(self.knowledge_retriever, self.document_retriever,
                         self.generator, self.memory, self.router)

    def run(self, question, has_document=False, general=False):
        if not has_document or self.document_retriever is None:
            return {"agent": "Report Generator Agent",
                    "answer": "Please upload a document first so I can generate a full report.",
                    "mode": "knowledge"}

        # gather outputs from the analysis agents
        clauses = self._mk(ClauseExtractionAgent).run("extract clauses", True)["answer"]
        risk = self._mk(RiskAnalysisAgent).run("risk analysis", True)["answer"]
        compliance = self._mk(ComplianceAgent).run("compliance check", True)["answer"]
        obligations = self._mk(ObligationAgent).run("obligations and deadlines", True)["answer"]

        sections = {
            "Key Clauses": clauses,
            "Risk Analysis": risk,
            "Compliance": compliance,
            "Obligations & Deadlines": obligations,
        }

        # export to PDF
        path = export_report_pdf(sections, filename="legal_report")

        answer = (
            "A full legal analysis report has been generated. It includes:\n"
            "- Key Clauses\n- Risk Analysis\n- Compliance Check\n- Obligations & Deadlines\n\n"
            f"Saved to: {path}"
        )
        self.memory.add_memory(question, answer)
        return {"agent": "Report Generator Agent", "answer": answer, "mode": "document"}
