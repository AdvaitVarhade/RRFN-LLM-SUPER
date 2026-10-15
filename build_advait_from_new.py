import docx
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn
import re, os, zipfile

def transform_report():
    input_path = 'paper/new.docx'
    output_path = 'project_report_Advait_Varhade_23BCE5098.docx'
    paper_output_path = 'paper/project_report_Advait_Varhade_23BCE5098.docx'
    
    print(f"Loading {input_path}...")
    doc = docx.Document(input_path)
    
    # -------------------------------------------------------------
    # 1. Update Front Matter for Advait Varhade
    # -------------------------------------------------------------
    print("Updating Front Matter for Advait Varhade...")
    
    # Paragraph 5: Cover page candidate list
    # Put Advait first
    doc.paragraphs[5].text = "ADVAIT RAVINDRAKUMAR VARHADE (23BCE5098)\nP. MANAS (23BCE5076)\nVIVAASH KOUNDAL (23BCE5095)"
    for r in doc.paragraphs[5].runs:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(13)
        r.bold = True
    doc.paragraphs[5].alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Paragraph 8: Cover page date
    doc.paragraphs[8].text = "October, 2026"
    for r in doc.paragraphs[8].runs:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
    doc.paragraphs[8].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Paragraph 14: Title page candidate list
    doc.paragraphs[14].text = "ADVAIT RAVINDRAKUMAR VARHADE (23BCE5098)\nP. MANAS (23BCE5076)\nVIVAASH KOUNDAL (23BCE5095)"
    for r in doc.paragraphs[14].runs:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(13)
        r.bold = True
    doc.paragraphs[14].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Paragraph 18: Title page date
    doc.paragraphs[18].text = "October, 2026"
    for r in doc.paragraphs[18].runs:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
    doc.paragraphs[18].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Paragraph 25: Declaration candidate signature
    doc.paragraphs[25].text = "Signature of the Candidate\nADVAIT RAVINDRAKUMAR VARHADE\n(Reg. No. 23BCE5098)"
    for r in doc.paragraphs[25].runs:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        r.bold = True
    doc.paragraphs[25].alignment = WD_ALIGN_PARAGRAPH.RIGHT

    # Certificate paragraph (Paragraph 29 in new.docx):
    # Change "submitted by P. MANAS (23BCE5076)" to "submitted by ADVAIT RAVINDRAKUMAR VARHADE (23BCE5098)"
    if "P. MANAS (23BCE5076)" in doc.paragraphs[29].text:
        doc.paragraphs[29].text = doc.paragraphs[29].text.replace("P. MANAS (23BCE5076)", "ADVAIT RAVINDRAKUMAR VARHADE (23BCE5098)")
        for r in doc.paragraphs[29].runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(12)

    # Acknowledgement:
    # Paragraph 42: Teammates mentioned
    doc.paragraphs[42].text = doc.paragraphs[42].text.replace("ADVAIT RAVINDRAKUMAR VARHADE (23BCE5098)", "P. MANAS (23BCE5076)")
    # Paragraph 44: Candidate signature
    doc.paragraphs[44].text = "ADVAIT RAVINDRAKUMAR VARHADE\n(Reg. No. 23BCE5098)"
    for r in doc.paragraphs[44].runs:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
        r.bold = True

    # -------------------------------------------------------------
    # 2. Update Section 1.6 Contributions Section (Table 6)
    # -------------------------------------------------------------
    print("Updating Section 1.6 Contributions Box (Table 6) for Advait Varhade...")
    t6 = doc.tables[6]
    cell = t6.rows[0].cells[0]
    
    advait_contrib_text = (
        "[INDIVIDUAL TECHNICAL CONTRIBUTION DECLARATION]\n"
        "CANDIDATE PRIMARY CONTRIBUTION FOCUS:\n"
        "This individual project report is submitted by ADVAIT RAVINDRAKUMAR VARHADE (23BCE5098) in fulfillment of the BCSE497J Project I requirements. While this report provides an end-to-end exposition of the complete RRFN-LLM-SUPER framework to preserve architectural coherence, the candidate's primary theoretical and engineering contributions focus on:\n\n"
        "Pillar: Mathematical Noise-Transition Modeling, Tikhonov Operator Inversion, Risk-Consistent Optimization & Theoretical Proofs\n"
        "Role: Lead Mathematical Modeler & Optimization Architect\n\n"
        "Key Technical Responsibilities:\n"
        "Mathematical formulation and formal proof of Theorem 3.1 (Spectral operator norm bound ||T_lambda^dagger||_2 <= 1/(2*sqrt(lambda)) and Lipschitz gradient stability), Theorem 3.2 (Surrogate risk consistency and excess clean risk bounds), confidence inversion resolvent error analysis, consensus anchor selection algorithm, transition matrix estimation, forward-corrected surrogate loss (L_risk), and software hardening of numerical linear algebra routines against singular matrix inversion.\n\n"
        "Core Modules Directly Authored / Implemented:\n"
        "  • rrfn/anchor_selector.py\n"
        "  • rrfn/transition_matrix.py\n"
        "  • rrfn/risk_loss.py\n"
        "  • data/attack_simulator.py\n"
        "  • tests/test_rrfn.py\n"
        "  • tests/test_attack_simulator.py\n"
        "  • tests/test_robustness_proofs.py"
    )
    cell.text = advait_contrib_text
    # Style runs in cell
    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)
    # Bold the header line
    cell.paragraphs[0].runs[0].bold = True

    # -------------------------------------------------------------
    # 3. Update Appendix 4 (Table 40 Plagiarism Statement)
    # -------------------------------------------------------------
    print("Updating Appendix 4 Plagiarism Box for Advait Varhade...")
    if len(doc.tables) > 40:
        t_plag = doc.tables[-1] # Last table is Appendix 4 plagiarism table
        for row in t_plag.rows:
            for c in row.cells:
                if "P. MANAS" in c.text:
                    c.text = c.text.replace("P. MANAS", "ADVAIT RAVINDRAKUMAR VARHADE")
                if "23BCE5076" in c.text:
                    c.text = c.text.replace("23BCE5076", "23BCE5098")
                for p in c.paragraphs:
                    for r in p.runs:
                        r.font.name = 'Times New Roman'

    # -------------------------------------------------------------
    # 4. Remove ALL Redundant Bullet Points and Numbered Points
    # -------------------------------------------------------------
    print("Cleaning redundant bullet points and numbered points across all chapters...")
    
    def clean_paragraph_runs(p):
        if not p.runs:
            return
        
        full_text = p.text.strip()
        
        # We only want to clean list items, bullets, or definition paragraphs that have duplicate titles/numbering.
        # DO NOT touch section headings like '1.1 BACKGROUND' or '3.4.1 VIEW 1'
        if p.style.name.startswith('Heading') or full_text.startswith('CHAPTER'):
            return
        # If it's an all-caps short heading like '1.1 BACKGROUND AND MOTIVATION'
        if re.match(r'^\d+\.\d+(?:\.\d+)?\s+[A-Z\s,–—\-]+$', full_text):
            return

        # Pattern A: '1. Random Attacks: Random Attacks: Bot profiles inject...'
        # Pattern B: '2. Average Attacks: Average Attacks: Filler item ratings...'
        # Pattern C: '3. Bandwagon Attacks: Bandwagon Shilling Attacks: The attacker...'
        # Pattern D: '4. Review-Bombing Attacks: Review-Bombing Nuke Attacks: Coordinated...'
        m = re.match(r'^(?:\d+\.|\([0-9]+\))\s*([^:]+):\s*(?:(?:(?:\d+\.|\([0-9]+\))\s*)?([^:]+):)?\s*(.*)$', full_text)
        if m and m.group(2):
            p1, p2, rest = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            # choose better title
            title = p2 if len(p2) >= len(p1) else p1
            # rebuild runs: run 0 bold title, run 1 body
            p.text = ""
            r0 = p.add_run(title + ": ")
            r0.bold = True
            r0.font.name = 'Times New Roman'
            r0.font.size = Pt(12)
            r1 = p.add_run(rest)
            r1.font.name = 'Times New Roman'
            r1.font.size = Pt(12)
            return

        # Pattern E: 'Issue 1: Spectral Instability: ...' -> 'Spectral Instability: ...'
        # 'Step 1: Catalog Partitioning: ...' -> 'Catalog Partitioning: ...'
        m = re.match(r'^(?:Issue|Step|Gap|Phase|Takeaway)\s+\d+:\s*(?:(?:Research Gap|Step|Issue|Phase|Takeaway)\s+\d+\s*)?(?:([^:]+):)?\s*(.*)$', full_text, re.IGNORECASE)
        if m and m.group(1):
            title = m.group(1).strip()
            rest = m.group(2).strip()
            p.text = ""
            r0 = p.add_run(title + ": ")
            r0.bold = True
            r0.font.name = 'Times New Roman'
            r0.font.size = Pt(12)
            r1 = p.add_run(rest)
            r1.font.name = 'Times New Roman'
            r1.font.size = Pt(12)
            return

        # Pattern F: 'Assumption 3.1: Assumption 3.1 (Anchor ...): ...'
        # 'Theorem 3.1: Theorem 3.1 (Spectral ...): ...'
        m = re.match(r'^(Assumption|Theorem)\s+(\d+\.\d+):\s*\1\s+\2\s*(\([^)]+\))?:\s*(.*)$', full_text, re.IGNORECASE)
        if m:
            kind, num, paren, rest = m.group(1), m.group(2), m.group(3) or '', m.group(4)
            title = f"{kind} {num} {paren}".strip()
            p.text = ""
            r0 = p.add_run(title + ": ")
            r0.bold = True
            r0.font.name = 'Times New Roman'
            r0.font.size = Pt(12)
            r1 = p.add_run(rest.strip())
            r1.font.name = 'Times New Roman'
            r1.font.size = Pt(12)
            return

        # Pattern G: '1. Minimizer Invariance: Part 1: ...' -> 'Minimizer Invariance: ...'
        # '1. Joint Denoising and Pareto Debiasing: Advancement 1: ...' -> 'Joint Denoising and Pareto Debiasing: ...'
        m = re.match(r'^(?:\d+\.|\([0-9]+\))\s*([^:]+):\s*(?:Part|Advancement)\s+\d+:\s*(.*)$', full_text, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
            rest = m.group(2).strip()
            p.text = ""
            r0 = p.add_run(title + ": ")
            r0.bold = True
            r0.font.name = 'Times New Roman'
            r0.font.size = Pt(12)
            r1 = p.add_run(rest)
            r1.font.name = 'Times New Roman'
            r1.font.size = Pt(12)
            return

        # Pattern H: '1. Data Pipeline: data: ...' -> 'Data Pipeline: ...'
        m = re.match(r'^(?:\d+\.|\([0-9]+\))\s*([^:]+):\s*[a-z_]+:\s*(.*)$', full_text)
        if m:
            title = m.group(1).strip()
            rest = m.group(2).strip()
            p.text = ""
            r0 = p.add_run(title + ": ")
            r0.bold = True
            r0.font.name = 'Times New Roman'
            r0.font.size = Pt(12)
            r1 = p.add_run(rest)
            r1.font.name = 'Times New Roman'
            r1.font.size = Pt(12)
            return

        # Pattern I: 'Tab 1: Tab 1 (Multi-View ...): ...' -> 'Tab 1 (Multi-View ...): ...'
        m = re.match(r'^Tab\s+(\d+):\s*Tab\s+\1\s*(\([^)]+\))?:\s*(.*)$', full_text, re.IGNORECASE)
        if m:
            paren = m.group(2) or ''
            title = f"Tab {m.group(1)} {paren}".strip()
            rest = m.group(3).strip()
            p.text = ""
            r0 = p.add_run(title + ": ")
            r0.bold = True
            r0.font.name = 'Times New Roman'
            r0.font.size = Pt(12)
            r1 = p.add_run(rest)
            r1.font.name = 'Times New Roman'
            r1.font.size = Pt(12)
            return

        # Pattern J: '1. Tikhonov Regularized Pseudo-Inversion: Numerical Inversion Guard: ...'
        # '1. Mathematical Noise Bounds & Risk Consistency: Theoretical Proofs: ...'
        m = re.match(r'^(?:\d+\.|\([0-9]+\))\s*([^:]+):\s*([^:]+):\s*(.*)$', full_text)
        if m:
            title = m.group(1).strip()
            rest = m.group(3).strip()
            p.text = ""
            r0 = p.add_run(title + ": ")
            r0.bold = True
            r0.font.name = 'Times New Roman'
            r0.font.size = Pt(12)
            r1 = p.add_run(rest)
            r1.font.name = 'Times New Roman'
            r1.font.size = Pt(12)
            return

        # Pattern K: Double bold prefixes across runs like:
        # run 0: 'Recall & nDCG: ', run 1: 'Recall@K & nDCG@K: Top-K ...'
        if len(p.runs) >= 2 and p.runs[0].bold:
            r0_txt = p.runs[0].text.strip()
            r1_txt = p.runs[1].text.strip()
            m_sub = re.match(r'^([^:]+):\s*(.*)$', r1_txt)
            if m_sub and r0_txt.endswith(':'):
                title0 = r0_txt[:-1].strip()
                title1 = m_sub.group(1).strip()
                # Check if titles are identical or highly similar
                if title0.lower() == title1.lower() or ('@' in title1 and title0.lower() in title1.lower()) or title0.lower() in title1.lower():
                    # Keep the more informative title1
                    p.runs[0].text = title1 + ": "
                    p.runs[1].text = m_sub.group(2)

    cleaned_bullets = 0
    for p in doc.paragraphs:
        clean_paragraph_runs(p)

    # -------------------------------------------------------------
    # 5. Save the updated Advait report
    # -------------------------------------------------------------
    print(f"Saving updated report to {output_path} and {paper_output_path}...")
    doc.save(output_path)
    doc.save(paper_output_path)
    
    # Also update master project_report.docx
    doc.save("project_report.docx")
    doc.save("paper/project_report.docx")
    print("SUCCESS: Advait Varhade report successfully updated!")

if __name__ == '__main__':
    transform_report()
