
# ---- Keyword signals (used for the keyword score) ----
DOC_TYPE_KEYWORDS = {
    "NDA": ["non-disclosure", "confidential information", "receiving party", "disclosing party"],
    "Lease Agreement": ["lease", "tenant", "landlord", "premises", "rent"],
    "Employment Contract": ["employment", "employee", "salary", "notice period", "employer"],
    "Service Agreement": ["service agreement", "provider", "scope of services", "deliverables"],
    "Vendor Agreement": ["vendor", "supplier", "purchase order", "goods"],
    "Privacy Policy": ["privacy policy", "personal data", "we collect", "cookies"],
    "Terms and Conditions": ["terms and conditions", "terms of use", "you agree"],
}

# ---- Semantic prototypes (used for the semantic score) ----
# Each type is described in plain language so the embedding model can
# match documents by MEANING even when the exact keywords are missing.
DOC_TYPE_PROTOTYPES = {
    "NDA":
        "A non-disclosure or confidentiality agreement between a disclosing party and a "
        "receiving party to keep secret or proprietary information private and not share it "
        "with any third party.",
    "Lease Agreement":
        "A lease or rental agreement between a landlord and a tenant for renting a house, flat, "
        "or premises, stating the monthly rent, security deposit, and lease term.",
    "Employment Contract":
        "An employment agreement where a company hires an individual for a job role, describing "
        "the salary or wage, working hours, leave, benefits, and notice period.",
    "Service Agreement":
        "A service agreement between a service provider and a client defining the scope of "
        "services, deliverables, payment terms, and project timelines.",
    "Vendor Agreement":
        "A vendor or supplier agreement for supplying goods or products, including a purchase "
        "order, delivery schedule, quality standards, and payment terms.",
    "Privacy Policy":
        "A privacy policy explaining how an organization collects, uses, stores, and protects "
        "the personal data of its users, including cookies and the user's right to delete data.",
    "Terms and Conditions":
        "Terms and conditions or terms of use that set the rules a user agrees to when using a "
        "website, application, or service.",
}
 