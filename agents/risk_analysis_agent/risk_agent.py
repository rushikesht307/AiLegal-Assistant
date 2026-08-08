from agents.base_agent import BaseLegalAgent


class RiskAnalysisAgent(BaseLegalAgent):

    AGENT_NAME = "Risk Analysis Agent"
    DEFAULT_QUERY = "contract risk analysis"
    SYSTEM_PROMPT = """
You are a senior legal contract risk analyst.

Your task is to conduct a professional legal risk assessment.

You MUST:

1. Review all contract provisions carefully.
2. Identify legal, commercial, operational, privacy, compliance,
   liability and financial risks.
3. Identify missing protections.
4. Evaluate severity for every issue.
5. Calculate an overall risk score.

Severity Scale:

- Low (10)
- Medium (25)
- High (50)
- Critical (75)

Risk Categories:

- Liability Risk
- Indemnity Risk
- Confidentiality Risk
- Data Privacy Risk
- Compliance Risk
- Payment Risk
- Termination Risk
- Intellectual Property Risk
- Operational Risk
- Regulatory Risk

For every identified risk provide:

Risk Title
Risk Category
Severity
Affected Party
Explanation
Business Impact
Recommendation

Then calculate:

Overall Risk Score (0-100)

Risk Rating:

0-25 = Low
26-50 = Medium
51-75 = High
76-100 = Critical

Finally provide:

1. Executive Summary
2. Key Risks
3. Missing Protections
4. Overall Risk Score
5. Overall Risk Rating
6. Recommendations

Only use information from the provided contract.
Do not invent clauses that are not present.
"""

    def run(
        self,
        question,
        has_document=False,
        general=False,
    ):

        mode, context = self.get_context(
            question,
            has_document,
            general,
        )

        risk_prompt = f"""
Perform a comprehensive legal risk analysis.

CONTRACT CONTENT:

{context}

USER REQUEST:

{question}

Instructions:

Step 1:
Review the contract completely.

Step 2:
Identify:

- Liability risks
- Indemnity risks
- Confidentiality risks
- Privacy risks
- Compliance risks
- Payment risks
- Termination risks
- Intellectual property risks
- Operational risks

Step 3:
Identify missing protections such as:

- Confidentiality clause
- Limitation of liability
- Governing law
- Dispute resolution
- Termination rights
- Data protection provisions

Step 4:
For each finding provide:

Risk Title:
Category:
Severity:
Affected Party:
Explanation:
Business Impact:
Recommendation:

Step 5:
Estimate a numerical score.

Low = 10
Medium = 25
High = 50
Critical = 75

Step 6:
Calculate:

Overall Risk Score (0-100)

and classify as:

Low
Medium
High
Critical

Step 7:
Generate the report in this format:

# Executive Summary

# Risk Findings

# Missing Protections

# Risk Score

# Recommendations
"""

        answer = self._ask_llm(
            risk_prompt,
            context,
        )

        self.memory.add_memory(
            question,
            answer,
        )

        return {
            "agent": self.AGENT_NAME,
            "answer": answer,
            "mode": mode,
        }