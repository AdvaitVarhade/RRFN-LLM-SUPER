"""
Front Matter Generator for BCSE497J Project Report
Contains Cover Page, Title Page, Declaration, Certificate, Abstract,
Acknowledgement, Table of Contents, List of Figures, List of Tables, and List of Acronyms.
Supports candidate-specific personalization adhering to VIT Chennai guidelines.
"""

import os
import docx
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from report_builder import (
    set_section_margins, set_section_page_numbering, add_footer_page_number,
    add_paragraph, add_callout_box
)
from student_configs import STUDENT_PROFILES

def get_default_student():
    return STUDENT_PROFILES[0]

def add_cover_page(doc, student_config=None):
    """
    Cover Page (Page 1): Formatted for the outer book cover.
    Includes 'A project report on' at the top, Title, Degree, all team members, VIT logo, School, and Date.
    Lists all team members, placing the candidate first as per VIT guidelines.
    """
    if student_config is None:
        student_config = get_default_student()

    p_top = doc.add_paragraph()
    p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_top.paragraph_format.space_before = Pt(14)
    p_top.paragraph_format.space_after = Pt(6)
    r_top = p_top.add_run("A project report on")
    r_top.font.name = 'Times New Roman'
    r_top.font.size = Pt(12)
    r_top.italic = True

    # Title of Project (TNR Font 20, Bold, CAPS)
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(8)
    p_title.paragraph_format.space_after = Pt(16)
    p_title.paragraph_format.line_spacing = 1.15
    r_title = p_title.add_run("ROBUST RISK-CONSISTENT GRAPH-CONTRASTIVE LEARNING FOR POPULARITY-CALIBRATED RECOMMENDATION WITH MIRF AUDITING")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(19)
    r_title.bold = True

    # Submitted in partial fulfillment... (TNR Font 14, Italic)
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(6)
    p_sub.paragraph_format.space_after = Pt(4)
    r_sub = p_sub.add_run("Submitted in partial fulfillment for the award of the degree of")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(13)
    r_sub.italic = True

    # Degree (TNR Font 22, Bold)
    p_deg = doc.add_paragraph()
    p_deg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_deg.paragraph_format.space_before = Pt(4)
    p_deg.paragraph_format.space_after = Pt(14)
    r_deg = p_deg.add_run("Bachelor of Technology in Computer Science and Engineering")
    r_deg.font.name = 'Times New Roman'
    r_deg.font.size = Pt(16)
    r_deg.bold = True

    # by (TNR Font 14, Italic)
    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_by.paragraph_format.space_before = Pt(4)
    p_by.paragraph_format.space_after = Pt(6)
    r_by = p_by.add_run("by")
    r_by.font.name = 'Times New Roman'
    r_by.font.size = Pt(13)
    r_by.italic = True

    # Candidate Names & Reg Nos (TNR Font 16, Bold, CAPS)
    # Mandated rule: Pages 1 & 2 list all team members
    p_cand = doc.add_paragraph()
    p_cand.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cand.paragraph_format.space_before = Pt(4)
    p_cand.paragraph_format.space_after = Pt(16)
    p_cand.paragraph_format.line_spacing = 1.25

    names = [(student_config["name"], student_config["reg_no"])]
    for tm in student_config["teammates"]:
        names.append((tm["name"], tm["reg_no"]))

    for idx, (name, reg) in enumerate(names):
        r_n = p_cand.add_run(f"{name} ({reg})")
        r_n.font.name = 'Times New Roman'
        r_n.font.size = Pt(13.5)
        r_n.bold = True
        if idx < len(names) - 1:
            p_cand.add_run("\n")

    # Logo
    p_logo = doc.add_paragraph()
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_logo.paragraph_format.space_before = Pt(8)
    p_logo.paragraph_format.space_after = Pt(12)
    logo_path = r"template_images/vit_logo.png"
    if os.path.exists(logo_path):
        p_logo.add_run().add_picture(logo_path, width=Cm(9.2))

    # School & Date
    p_sch = doc.add_paragraph()
    p_sch.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sch.paragraph_format.space_before = Pt(6)
    p_sch.paragraph_format.space_after = Pt(3)
    r_sch = p_sch.add_run("SCHOOL OF COMPUTER SCIENCE AND ENGINEERING")
    r_sch.font.name = 'Times New Roman'
    r_sch.font.size = Pt(14)
    r_sch.bold = True

    p_dt = doc.add_paragraph()
    p_dt.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dt.paragraph_format.space_before = Pt(2)
    p_dt.paragraph_format.space_after = Pt(6)
    r_dt = p_dt.add_run("November, 2025")
    r_dt.font.name = 'Times New Roman'
    r_dt.font.size = Pt(12)

    doc.add_page_break()

def add_title_page(doc, student_config=None):
    """
    Title Page (Page 2): Formatted for the inner title sheet of the bound report.
    As per template: starts directly with TITLE OF THE PROJECT REPORT (omits 'A project report on').
    """
    if student_config is None:
        student_config = get_default_student()

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(24)
    p_title.paragraph_format.space_after = Pt(14)
    p_title.paragraph_format.line_spacing = 1.15
    r_title = p_title.add_run("ROBUST RISK-CONSISTENT GRAPH-CONTRASTIVE LEARNING FOR POPULARITY-CALIBRATED RECOMMENDATION WITH MIRF AUDITING")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(19)
    r_title.bold = True

    # Submitted in partial fulfillment... (TNR Font 14, Italic)
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(6)
    p_sub.paragraph_format.space_after = Pt(4)
    r_sub = p_sub.add_run("Submitted in partial fulfillment for the award of the degree of")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(13)
    r_sub.italic = True

    # Degree (TNR Font 22, Bold)
    p_deg = doc.add_paragraph()
    p_deg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_deg.paragraph_format.space_before = Pt(4)
    p_deg.paragraph_format.space_after = Pt(14)
    r_deg = p_deg.add_run("Bachelor of Technology in Computer Science and Engineering")
    r_deg.font.name = 'Times New Roman'
    r_deg.font.size = Pt(16)
    r_deg.bold = True

    # by (TNR Font 14, Italic)
    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_by.paragraph_format.space_before = Pt(4)
    p_by.paragraph_format.space_after = Pt(6)
    r_by = p_by.add_run("by")
    r_by.font.name = 'Times New Roman'
    r_by.font.size = Pt(13)
    r_by.italic = True

    # Candidate Names & Reg Nos (TNR Font 16, Bold, CAPS)
    p_cand = doc.add_paragraph()
    p_cand.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cand.paragraph_format.space_before = Pt(4)
    p_cand.paragraph_format.space_after = Pt(12)
    p_cand.paragraph_format.line_spacing = 1.25

    names = [(student_config["name"], student_config["reg_no"])]
    for tm in student_config["teammates"]:
        names.append((tm["name"], tm["reg_no"]))

    for idx, (name, reg) in enumerate(names):
        r_n = p_cand.add_run(f"{name} ({reg})")
        r_n.font.name = 'Times New Roman'
        r_n.font.size = Pt(13.5)
        r_n.bold = True
        if idx < len(names) - 1:
            p_cand.add_run("\n")

    # Guide details on Title Page
    p_gd = doc.add_paragraph()
    p_gd.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_gd.paragraph_format.space_before = Pt(4)
    p_gd.paragraph_format.space_after = Pt(6)
    r_gdh = p_gd.add_run("Under the guidance of\n")
    r_gdh.font.name = 'Times New Roman'
    r_gdh.font.size = Pt(12)
    r_gdh.italic = True
    r_gdn = p_gd.add_run("Dr. Suganeshwari G\nAssociate Professor")
    r_gdn.font.name = 'Times New Roman'
    r_gdn.font.size = Pt(13)
    r_gdn.bold = True

    # Logo
    p_logo = doc.add_paragraph()
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_logo.paragraph_format.space_before = Pt(6)
    p_logo.paragraph_format.space_after = Pt(10)
    logo_path = r"template_images/vit_logo.png"
    if os.path.exists(logo_path):
        p_logo.add_run().add_picture(logo_path, width=Cm(8.8))

    # School & Date
    p_sch = doc.add_paragraph()
    p_sch.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sch.paragraph_format.space_before = Pt(4)
    p_sch.paragraph_format.space_after = Pt(2)
    r_sch = p_sch.add_run("SCHOOL OF COMPUTER SCIENCE AND ENGINEERING")
    r_sch.font.name = 'Times New Roman'
    r_sch.font.size = Pt(14)
    r_sch.bold = True

    p_dt = doc.add_paragraph()
    p_dt.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dt.paragraph_format.space_before = Pt(2)
    p_dt.paragraph_format.space_after = Pt(6)
    r_dt = p_dt.add_run("November, 2025")
    r_dt.font.name = 'Times New Roman'
    r_dt.font.size = Pt(12)

    doc.add_page_break()

def add_declaration_page(doc, student_config=None):
    """
    Declaration Page: Mandated rule: On the Declaration page, mention ONLY individual name.
    """
    if student_config is None:
        student_config = get_default_student()

    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_h.paragraph_format.space_before = Pt(18)
    p_h.paragraph_format.space_after = Pt(24)
    r_h = p_h.add_run("DECLARATION")
    r_h.font.name = 'Times New Roman'
    r_h.font.size = Pt(14)
    r_h.bold = True
    r_h.underline = True

    add_paragraph(
        doc,
        f"I hereby declare that the thesis entitled “ROBUST RISK-CONSISTENT GRAPH-CONTRASTIVE LEARNING FOR POPULARITY-CALIBRATED RECOMMENDATION WITH MIRF AUDITING” "
        f"submitted by {student_config['name']} ({student_config['reg_no']}), for the award of the degree of Bachelor of Technology in Computer Science and Engineering, "
        f"Vellore Institute of Technology, Chennai is a record of bonafide work carried out by me under the supervision of Dr. Suganeshwari G, Associate Professor, "
        f"School of Computer Science and Engineering, Vellore Institute of Technology, Chennai."
    )

    add_paragraph(
        doc,
        "I further declare that the work reported in this thesis has not been submitted and will not be submitted, either in part or in full, "
        "for the award of any other degree or diploma in this institute or any other institute or university."
    )

    # Place, Date, Signature
    p_sign = doc.add_paragraph()
    p_sign.paragraph_format.space_before = Pt(48)
    p_sign.paragraph_format.space_after = Pt(6)
    p_sign.paragraph_format.line_spacing = 1.3
    
    r_pl = p_sign.add_run("Place: Chennai\nDate: November 2025")
    r_pl.font.name = 'Times New Roman'
    r_pl.font.size = Pt(12)
    r_pl.bold = True
    
    # Signature on right
    p_cand = doc.add_paragraph()
    p_cand.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_cand.paragraph_format.space_before = Pt(16)
    p_cand.paragraph_format.space_after = Pt(20)
    r_cand = p_cand.add_run(f"Signature of the Candidate\n{student_config['name']}\n(Reg. No. {student_config['reg_no']})")
    r_cand.font.name = 'Times New Roman'
    r_cand.font.size = Pt(12)
    r_cand.bold = True

    doc.add_page_break()

def add_certificate_page(doc, student_config=None):
    """
    Certificate Page: Mandated rule: On the Certificate page, mention ONLY individual name.
    """
    if student_config is None:
        student_config = get_default_student()

    p_sch = doc.add_paragraph()
    p_sch.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sch.paragraph_format.space_before = Pt(12)
    p_sch.paragraph_format.space_after = Pt(4)
    r_sch = p_sch.add_run("School of Computer Science and Engineering")
    r_sch.font.name = 'Times New Roman'
    r_sch.font.size = Pt(14)
    r_sch.bold = True

    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_h.paragraph_format.space_before = Pt(4)
    p_h.paragraph_format.space_after = Pt(20)
    r_h = p_h.add_run("CERTIFICATE")
    r_h.font.name = 'Times New Roman'
    r_h.font.size = Pt(14)
    r_h.bold = True
    r_h.underline = True

    add_paragraph(
        doc,
        f"This is to certify that the report entitled “ROBUST RISK-CONSISTENT GRAPH-CONTRASTIVE LEARNING FOR POPULARITY-CALIBRATED RECOMMENDATION WITH MIRF AUDITING” "
        f"is prepared and submitted by {student_config['name']} ({student_config['reg_no']}) to Vellore Institute of Technology, Chennai, "
        f"in partial fulfillment of the requirement for the award of the degree of Bachelor of Technology in Computer Science and Engineering "
        f"is a bonafide record of work carried out under my guidance. The project fulfills the requirements as per the regulations of this University "
        f"and in my opinion meets the necessary academic and technical standards for submission. The contents of this report have not been submitted "
        f"and will not be submitted either in part or in full, for the award of any other degree or diploma and the same is certified."
    )

    # Signatures Table
    tbl = doc.add_table(rows=4, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    for r in tbl.rows:
        r.cells[0].width = Cm(7.2)
        r.cells[1].width = Cm(7.2)
        
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="none"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="none"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tbl._tbl.tblPr.append(borders)

    # Row 0: Guide
    c00 = tbl.rows[0].cells[0]
    p00 = c00.paragraphs[0]
    p00.paragraph_format.space_before = Pt(16)
    p00.paragraph_format.space_after = Pt(2)
    p00.paragraph_format.line_spacing = 1.2
    r00 = p00.add_run("Signature of the Project Guide:\nName: Dr. Suganeshwari G\nDesignation: Associate Professor\nDate: November 2025")
    r00.font.name = 'Times New Roman'
    r00.font.size = Pt(11)
    r00.bold = True

    # Row 1: Examiners
    c10 = tbl.rows[1].cells[0]
    p10 = c10.paragraphs[0]
    p10.paragraph_format.space_before = Pt(20)
    p10.paragraph_format.space_after = Pt(2)
    p10.paragraph_format.line_spacing = 1.2
    r10 = p10.add_run("Signature of the Internal Examiner:\nName:\nDate:")
    r10.font.name = 'Times New Roman'
    r10.font.size = Pt(11)
    r10.bold = True

    c11 = tbl.rows[1].cells[1]
    p11 = c11.paragraphs[0]
    p11.paragraph_format.space_before = Pt(20)
    p11.paragraph_format.space_after = Pt(2)
    p11.paragraph_format.line_spacing = 1.2
    r11 = p11.add_run("Signature of the External Examiner:\nName:\nDate:")
    r11.font.name = 'Times New Roman'
    r11.font.size = Pt(11)
    r11.bold = True

    # Row 2: HoD Approval
    c20 = tbl.rows[2].cells[0]
    p20 = c20.paragraphs[0]
    p20.paragraph_format.space_before = Pt(20)
    p20.paragraph_format.space_after = Pt(2)
    p20.paragraph_format.line_spacing = 1.2
    r20 = p20.add_run("Approved by the Head of Department:\nDepartment of Computer Science and Engineering\nName: Head of the Department\nDate:")
    r20.font.name = 'Times New Roman'
    r20.font.size = Pt(11)
    r20.bold = True

    # Row 3: School Seal placeholder
    c31 = tbl.rows[3].cells[1]
    p31 = c31.paragraphs[0]
    p31.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p31.paragraph_format.space_before = Pt(16)
    p31.paragraph_format.space_after = Pt(2)
    r31 = p31.add_run("[ Seal of School of Computer Science and Engineering ]")
    r31.font.name = 'Times New Roman'
    r31.font.size = Pt(10)
    r31.italic = True

    doc.add_page_break()

def add_abstract_page(doc, student_config=None):
    """
    Abstract Page: Strictly one page only. Unified across all candidates.
    """
    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_h.paragraph_format.space_before = Pt(6)
    p_h.paragraph_format.space_after = Pt(8)
    r_h = p_h.add_run("ABSTRACT")
    r_h.font.name = 'Times New Roman'
    r_h.font.size = Pt(14)
    r_h.bold = True
    r_h.underline = True

    def _add_abs_p(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
        return p

    _add_abs_p(
        "Modern web-scale recommender systems rely fundamentally on bipartite user-item interaction graphs to deliver personalized recommendations across digital platforms. "
        "However, real-world deployment graphs suffer from two concurrent, orthogonal systemic vulnerabilities: (1) adversarial data poisoning via bandwagon shilling attacks "
        "and coordinated review-bombing bursts designed to inject deceptive feedback, and (2) structural catalog popularity bias where Pareto-skewed interaction "
        "distributions starve authentic long-tail items of visibility. Existing paradigms treat adversarial robustness and popularity debiasing as isolated objectives, "
        "collapsing catastrophically when both co-occur: debiasing models mistake injected bot interactions for genuine niche enthusiasm, aggressively promoting spam targets "
        "into recommended long-tail quotas."
    )

    _add_abs_p(
        "To resolve this fatal concurrence, this project introduces RRFN-LLM-SUPER, an end-to-end framework integrating stochastic noise-transition modeling, multi-modal "
        "interaction reliability filtering, risk-aware graph contrastive learning, and popularity-calibrated blueprint debiasing. We theoretically prove a spectral operator norm "
        "upper bound for Tikhonov-regularized noise inversion (||T_lambda^dagger||_2 <= 1 / (2*sqrt(lambda))), establishing Lipschitz gradient stability and forward surrogate excess risk consistency. "
        "To audit edge authenticity, we architect the Multi-view Interaction Reliability Filter (MIRF), synthesizing model prediction consistency, asynchronous Large Language Model (LLM) "
        "review text verification supported by a thread-safe SQLite WAL caching layer, and hardware-accelerated 2D BLAS review-bombing burst detection executing in sub-5 ms (>500x speedup). "
        "The resulting sample reliability weights (w_ui in [0, 1]) modulate InfoNCE contrastive alignment across SimGCL and SGL backbones, and purify historical user blueprints for calibrated long-tail exposure."
    )

    _add_abs_p(
        "Extensive 5-seed empirical benchmarks (Seeds 42–46) across MovieLens-1M, Amazon Electronics, and Yelp Academic validate our framework. RRFN-LLM-SUPER achieves an Attack Detection AP "
        "of 0.6884 on Amazon and 0.6816 on Yelp (a 3.9–4.3x gain over random baselines), maintains an Authentic Retention AP of 0.9989 (ROC-AUC = 0.7931 under 15% bandwagon shilling), "
        "and achieves statistically significant improvements in popularity calibration error (RMSE-PC 0.1067, p = 0.0166; MRMC 0.1128, p = 0.0088). Furthermore, blueprint purification successfully "
        "repels deceptive tail inflation, reducing spam target exposure (STER@10) from 88% to 72% while elevating genuine long-tail coverage (g-LTC@10 = 0.4202 vs. 0.3371, p = 0.0397 on Yelp). "
        "A 1,000-sample Monte Carlo economic simulation further confirms significant platform revenue protection (p < 0.001)."
    )

    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_before = Pt(4)
    p_kw.paragraph_format.space_after = Pt(0)
    r_kwh = p_kw.add_run("Keywords: ")
    r_kwh.font.name = 'Times New Roman'
    r_kwh.font.size = Pt(10.5)
    r_kwh.bold = True
    r_kwt = p_kw.add_run("Recommender Systems, Graph Neural Networks, Graph Contrastive Learning, Multi-view Interaction Reliability Filter (MIRF), "
                         "Review-Bombing Detection, Popularity Bias, Long-Tail Calibration, Large Language Models, Trustworthy AI, Risk Consistency.")
    r_kwt.font.name = 'Times New Roman'
    r_kwt.font.size = Pt(10.5)
    r_kwt.italic = True

    doc.add_page_break()

def add_acknowledgement_page(doc, student_config=None):
    """
    Acknowledgement Page: Mandated rule: On the Acknowledgement page, mention ONLY your individual name.
    """
    if student_config is None:
        student_config = get_default_student()

    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_h.paragraph_format.space_before = Pt(8)
    p_h.paragraph_format.space_after = Pt(8)
    r_h = p_h.add_run("ACKNOWLEDGEMENT")
    r_h.font.name = 'Times New Roman'
    r_h.font.size = Pt(14)
    r_h.bold = True

    def _add_ack_p(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
        return p

    _add_ack_p(
        "It is my utmost pleasure to express with a deep sense of gratitude to Dr. Suganeshwari G, Associate Professor, "
        "School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, for her constant guidance, continual encouragement, "
        "understanding, and intellectual mentorship throughout this endeavor. Her deep insights and rigorous critique in the domains of "
        "Recommender Systems, Graph Machine Learning, and Trustworthy AI have been invaluable in shaping the theoretical and experimental dimensions of this research."
    )

    _add_ack_p(
        "It is with profound gratitude that I extend my heartfelt thanks to the visionary leadership of Vellore Institute of Technology: "
        "Dr. G. Viswanathan, our Honorable Chancellor; Mr. Sankar Viswanathan, Dr. Sekar Viswanathan, and Dr. G. V. Selvam, Vice Presidents; "
        "Dr. Sandhya Pentareddy, Executive Director; Ms. Kadhambari S. Viswanathan, Assistant Vice-President; "
        "Dr. V. S. Kanchana Bhaaskaran, Vice-Chancellor; Dr. T. Thyagarajan, Pro-Vice Chancellor, VIT Chennai; and Dr. P. K. Manoharan, Additional Registrar, "
        "for providing an exceptional academic environment, cutting-edge computing infrastructure, and continual inspiration during the course of my study."
    )

    _add_ack_p(
        "Special mention and sincere thanks are extended to Dr. Viswanathan V, Dean; Dr. Nithyanandam P, Associate Dean; "
        "Dr. Suganya G, Associate Dean; and Dr. Sweetlin Hemalatha C, Associate Dean, School of Computer Science and Engineering, "
        "Vellore Institute of Technology, Chennai, for spending their valuable time and efforts in sharing their academic knowledge and supporting us in every phase."
    )

    _add_ack_p(
        "In a jubilant state, I express my whole-hearted thanks to the Head of the Department, Department of Computer Science and Engineering, "
        "as well as the BCSE497J Project Coordinators and faculty committee for their structured academic framework, periodic reviews, and constant encouragement."
    )

    # Teammate acknowledgement
    tm_mentions = []
    for tm in student_config["teammates"]:
        tm_mentions.append(f"{tm['name']} ({tm['reg_no']}) for his collaborative partnership on {tm['pillar']}")
    tm_text = "; and ".join(tm_mentions)

    _add_ack_p(
        f"I also extend my sincere appreciation to my project teammates: {tm_text}, for their camaraderie, constructive technical brainstorming, "
        f"and collaborative synergy throughout the execution of this thesis. My deepest and most affectionate gratitude goes to my parents and family "
        f"members for their unconditional sacrifices, moral backing, and prayers. Finally, I thank all my friends and peers at VIT Chennai who supported me throughout this journey."
    )

    p_sign = doc.add_paragraph()
    p_sign.paragraph_format.space_before = Pt(12)
    p_sign.paragraph_format.space_after = Pt(2)
    p_sign.paragraph_format.line_spacing = 1.15
    
    r_pl = p_sign.add_run("Place: Chennai\nDate: November 2025")
    r_pl.font.name = 'Times New Roman'
    r_pl.font.size = Pt(11)
    r_pl.bold = True
    
    p_cand = doc.add_paragraph()
    p_cand.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_cand.paragraph_format.space_before = Pt(2)
    p_cand.paragraph_format.space_after = Pt(4)
    r_cand = p_cand.add_run(f"{student_config['name']}\n(Reg. No. {student_config['reg_no']})")
    r_cand.font.name = 'Times New Roman'
    r_cand.font.size = Pt(11)
    r_cand.bold = True

    doc.add_page_break()

def add_table_of_contents_page(doc, student_config=None):
    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_h.paragraph_format.space_before = Pt(18)
    p_h.paragraph_format.space_after = Pt(18)
    r_h = p_h.add_run("CONTENTS")
    r_h.font.name = 'Times New Roman'
    r_h.font.size = Pt(16)
    r_h.bold = True

    toc_items = [
        ("DECLARATION", "iii"),
        ("CERTIFICATE", "iv"),
        ("ABSTRACT", "v"),
        ("ACKNOWLEDGEMENT", "vi"),
        ("CONTENTS", "vii"),
        ("LIST OF FIGURES", "ix"),
        ("LIST OF TABLES", "xi"),
        ("LIST OF ACRONYMS", "xii"),
        ("CHAPTER 1   INTRODUCTION", "1"),
        ("   1.1  Background and Motivation", "1"),
        ("   1.2  Systemic Vulnerabilities in Graph Recommenders", "3"),
        ("   1.3  Problem Statement and Formal Challenges", "5"),
        ("   1.4  Objectives of the Project", "6"),
        ("   1.5  Grounding Key Concepts: SUP, RRFN, and MIRF", "7"),
        ("   1.6  Project Work Division and Technical Contributions Matrix", "9"),
        ("   1.7  Scope and Boundaries of the Work", "10"),
        ("   1.8  Organization of the Project Report", "11"),
        ("CHAPTER 2   LITERATURE REVIEW AND THEORETICAL BACKGROUND", "13"),
        ("   2.1  Overview of Collaborative Filtering Paradigms", "13"),
        ("   2.2  Adversarial Shilling Attacks in Recommender Systems", "15"),
        ("   2.3  Noise-Transition Modeling and Loss Correction", "17"),
        ("   2.4  Catalog Popularity Skew and Long-Tail Calibration", "19"),
        ("   2.5  Soft Blueprint Merging (SUPER/SUP) and Vulnerabilities", "21"),
        ("   2.6  Graph Contrastive Learning (GCL: SGL, SimGCL)", "23"),
        ("   2.7  Trustworthy Graph Learning & Positioning vs. Trust-GRS", "25"),
        ("   2.8  Large Language Models (LLMs) in Recommendation Auditing", "27"),
        ("   2.9  Critical Research Gaps in Prior Literature", "28"),
        ("   2.10 Comprehensive Literature Positioning Matrix (Table 2.1)", "29"),
        ("CHAPTER 3   SYSTEM DESIGN AND MATHEMATICAL METHODOLOGY", "31"),
        ("   3.1  End-to-End System Architecture (Figure 3.1)", "31"),
        ("   3.2  Mathematical Formulations and Noise Transition Model", "33"),
        ("   3.3  Stage 1: Anchor Selection & Tikhonov Noise Inversion", "35"),
        ("   3.4  Stage 2: Multi-View Interaction Reliability Filter (MIRF)", "37"),
        ("       3.4.1 Statistical Prediction Consistency (R_RRFN)", "37"),
        ("       3.4.2 LLM Review Auditing with SQLite WAL Cache (R_LLM)", "38"),
        ("       3.4.3 2D BLAS Vectorized Review-Bombing Burst Detection", "39"),
        ("       3.4.4 Continuous Signal Synthesis and Trust Weights (w_ui)", "41"),
        ("   3.5  Stage 3: Risk-Aware Graph Contrastive Learning", "42"),
        ("       3.5.1 SimGCL Latent Perturbations & SGL Structural Dropout", "42"),
        ("       3.5.2 Sample-Weighted InfoNCE Contrastive Objective", "43"),
        ("       3.5.3 Dual Partition Training (Head and Tail Models)", "44"),
        ("   3.6  Stage 4: Denoised SUPER Blueprint & Trajectory Interleaving", "45"),
        ("   3.7  Theoretical Proofs and Formal Mathematical Guarantees", "47"),
        ("       3.7.1 Theorem 1: Spectral Operator Norm Bound Proof", "47"),
        ("       3.7.2 Theorem 2: Surrogate Excess Risk Consistency Proof", "49"),
        ("       3.7.3 Confidence Inversion Resolvent Error Bounds", "51"),
        ("   3.8  Formal Definitions of Recommendation and Detection Metrics", "52"),
        ("CHAPTER 4   SYSTEM IMPLEMENTATION AND ARCHITECTURE DETAILS", "54"),
        ("   4.1  Software Architecture and Design Principles", "54"),
        ("   4.2  Comprehensive Module Mapping and File Index (Table 4.1)", "55"),
        ("   4.3  Multi-Domain Data Ingestion and Robust Loading Pipelines", "57"),
        ("   4.4  Graph Neural Network Backbones and Contrastive Engine", "59"),
        ("   4.5  MIRF Filter Modules and High-Performance BLAS Acceleration", "61"),
        ("   4.6  Denoised SUPER Calibrated Recommender Engine", "63"),
        ("   4.7  Algorithmic Workflows and End-to-End Pseudocode", "65"),
        ("   4.8  Interactive Telemetry and Streamlit Web Dashboard", "67"),
        ("   4.9  Software Hardening, Numerical Protection, and Concurrency", "69"),
        ("CHAPTER 5   EXPERIMENTAL EVALUATION, RESULTS, AND DISCUSSION", "71"),
        ("   5.1  Experimental Setup and Dataset Characteristics (Table 5.1)", "71"),
        ("   5.2  Adversarial Attack Simulation Protocols", "73"),
        ("   5.3  Multi-Seed Statistical Validation Protocol (Seeds 42-46)", "75"),
        ("   5.4  Empirical Benchmark Results on MovieLens-1M (Table 5.2)", "76"),
        ("   5.5  Multi-Domain Benchmarking on Amazon and Yelp (Table 5.3)", "78"),
        ("   5.6  MIRF Attack Detection vs. Authentic Retention Analysis", "80"),
        ("   5.7  Component Ablation Study and Hyperparameter Sweep (Table 5.4)", "82"),
        ("   5.8  Dual Partition Model Convergence and Training Loss Curves", "84"),
        ("   5.9  Scenario-Based Offline Economic Sensitivity Simulation", "85"),
        ("   5.10 Discussion of Engineering Trade-Offs and System Limitations", "87"),
        ("CHAPTER 6   CONCLUSION AND FUTURE WORK", "89"),
        ("   6.1  Summary of Completed Work and Contributions", "89"),
        ("   6.2  Key Academic and Practical Findings", "90"),
        ("   6.3  Realistic System Limitations", "91"),
        ("   6.4  Future Research Avenues and Extensions", "92"),
        ("REFERENCES (APA FORMAT)", "94"),
        ("APPENDIX A   REPOSITORY ARCHITECTURE AND COMPLETE SOURCE TREE", "97"),
        ("APPENDIX B   DETAILED HYPERPARAMETER CONFIGURATIONS (Table B.1)", "99"),
        ("APPENDIX C   SELECTED CORE ALGORITHMIC IMPLEMENTATION LISTINGS", "100"),
        ("APPENDIX D   PLAGIARISM CHECK COMPLIANCE AND SIMILARITY REPORT", "102")
    ]

    tbl = doc.add_table(rows=len(toc_items) + 1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="2E5B82"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="2E5B82"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="none"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tbl._tbl.tblPr.append(borders)
    
    # Header
    hdr = tbl.rows[0]
    hdr.cells[0].width = Cm(12.5)
    hdr.cells[1].width = Cm(2.0)
    
    shd0 = parse_xml(f'<w:shd {nsdecls("w")} w:fill="EBF2FA"/>')
    shd1 = parse_xml(f'<w:shd {nsdecls("w")} w:fill="EBF2FA"/>')
    hdr.cells[0]._tc.get_or_add_tcPr().append(shd0)
    hdr.cells[1]._tc.get_or_add_tcPr().append(shd1)
    
    p0 = hdr.cells[0].paragraphs[0]
    p0.paragraph_format.space_before = Pt(4)
    p0.paragraph_format.space_after = Pt(4)
    r0 = p0.add_run("TOPIC / CHAPTER TITLE")
    r0.font.name = 'Times New Roman'
    r0.font.size = Pt(11)
    r0.bold = True
    
    p1 = hdr.cells[1].paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p1.paragraph_format.space_before = Pt(4)
    p1.paragraph_format.space_after = Pt(4)
    r1 = p1.add_run("PAGE")
    r1.font.name = 'Times New Roman'
    r1.font.size = Pt(11)
    r1.bold = True

    for idx, (title, pg) in enumerate(toc_items):
        row = tbl.rows[idx + 1]
        row.cells[0].width = Cm(12.5)
        row.cells[1].width = Cm(2.0)
        
        pt = row.cells[0].paragraphs[0]
        pt.paragraph_format.space_before = Pt(2)
        pt.paragraph_format.space_after = Pt(2)
        pt.paragraph_format.line_spacing = 1.05
        rt = pt.add_run(title)
        rt.font.name = 'Times New Roman'
        rt.font.size = Pt(10)
        if title.startswith("CHAPTER") or title in ["DECLARATION", "CERTIFICATE", "ABSTRACT", "ACKNOWLEDGEMENT", "CONTENTS", "LIST OF FIGURES", "LIST OF TABLES", "LIST OF ACRONYMS", "REFERENCES (APA FORMAT)"]:
            rt.bold = True
            
        pp = row.cells[1].paragraphs[0]
        pp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        pp.paragraph_format.space_before = Pt(2)
        pp.paragraph_format.space_after = Pt(2)
        pp.paragraph_format.line_spacing = 1.05
        rp = pp.add_run(pg)
        rp.font.name = 'Times New Roman'
        rp.font.size = Pt(10)
        if title.startswith("CHAPTER") or title in ["DECLARATION", "CERTIFICATE", "ABSTRACT", "ACKNOWLEDGEMENT", "CONTENTS", "LIST OF FIGURES", "LIST OF TABLES", "LIST OF ACRONYMS"]:
            rp.bold = True

    doc.add_page_break()

def add_list_of_figures_page(doc):
    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_h.paragraph_format.space_before = Pt(18)
    p_h.paragraph_format.space_after = Pt(18)
    r_h = p_h.add_run("LIST OF FIGURES")
    r_h.font.name = 'Times New Roman'
    r_h.font.size = Pt(16)
    r_h.bold = True

    fig_items = [
        ("Figure 3.1", "End-to-End System Architecture of the RRFN-LLM-SUPER Framework", "32"),
        ("Figure 3.2", "Detailed Multi-View Interaction Reliability Filter (MIRF) Tri-Signal Pipeline", "38"),
        ("Figure 3.3", "Denoised SUPER Blueprint Assembly and Calibrated Trajectory Interleaving Flow", "46"),
        ("Figure 4.1", "Interactive Streamlit Web Dashboard: 4-Panel Live Telemetry & Inspector", "68"),
        ("Figure 5.1", "Empirical ROC Curve and Precision-Recall Curves for Attack & Authentic Detection", "80"),
        ("Figure 5.2", "Sensitivity Sweep of MIRF Anomaly Weights Along the Simplices omega_1 + omega_2 = 1.0", "83"),
        ("Figure 5.3", "Dual Partition Head/Tail Training Convergence and InfoNCE Contrastive Loss Curves", "84"),
        ("Figure 5.4", "Multi-Domain Radar Comparison Across MovieLens-1M, Amazon Reviews, and Yelp Academic", "79")
    ]

    tbl = doc.add_table(rows=len(fig_items) + 1, cols=3)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="2E5B82"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="2E5B82"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="none"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tbl._tbl.tblPr.append(borders)
    
    hdr = tbl.rows[0]
    hdr.cells[0].width = Cm(2.8)
    hdr.cells[1].width = Cm(9.7)
    hdr.cells[2].width = Cm(2.0)
    
    for c_i, h_txt in enumerate(["FIGURE NO.", "CAPTION / DESCRIPTION", "PAGE"]):
        cell = hdr.cells[c_i]
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="EBF2FA"/>')
        cell._tc.get_or_add_tcPr().append(shd)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        if c_i == 2:
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = p.add_run(h_txt)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
        r.bold = True

    for idx, (f_no, f_desc, f_pg) in enumerate(fig_items):
        row = tbl.rows[idx + 1]
        row.cells[0].width = Cm(2.8)
        row.cells[1].width = Cm(9.7)
        row.cells[2].width = Cm(2.0)
        
        p0 = row.cells[0].paragraphs[0]
        p0.paragraph_format.space_before = Pt(3)
        p0.paragraph_format.space_after = Pt(3)
        r0 = p0.add_run(f_no)
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(10)
        r0.bold = True
        
        p1 = row.cells[1].paragraphs[0]
        p1.paragraph_format.space_before = Pt(3)
        p1.paragraph_format.space_after = Pt(3)
        r1 = p1.add_run(f_desc)
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(10)
        
        p2 = row.cells[2].paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p2.paragraph_format.space_before = Pt(3)
        p2.paragraph_format.space_after = Pt(3)
        r2 = p2.add_run(f_pg)
        r2.font.name = 'Times New Roman'
        r2.font.size = Pt(10)
        r2.bold = True

    doc.add_page_break()

def add_list_of_tables_page(doc):
    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_h.paragraph_format.space_before = Pt(18)
    p_h.paragraph_format.space_after = Pt(18)
    r_h = p_h.add_run("LIST OF TABLES")
    r_h.font.name = 'Times New Roman'
    r_h.font.size = Pt(16)
    r_h.bold = True

    tbl_items = [
        ("Table 1.1", "Comprehensive Project Work Division and Technical Contribution Matrix", "10"),
        ("Table 2.1", "Comprehensive Literature Positioning Matrix: Adversarial Robustness vs. Debiasing", "29"),
        ("Table 3.1", "Detailed 14-Dimension Architectural Comparison (Standalone SUP vs. RRFN-LLM-SUPER)", "34"),
        ("Table 4.1", "Complete Software Architecture, Module Responsibility, and File System Index", "56"),
        ("Table 5.1", "Dataset Characteristics, Graph Topology, and Bandwagon Attack Injection Statistics", "72"),
        ("Table 5.2", "Empirical 5-Seed Benchmark Results on MovieLens-1M Sub-Graph under 15% Attack", "77"),
        ("Table 5.3", "Multi-Domain 5-Seed Benchmark on Amazon Reviews (SimGCL) and Yelp Academic (SGL)", "79"),
        ("Table 5.4", "Comprehensive Component Ablation Study on Amazon Reviews (SimGCL) under Attack", "82"),
        ("Table B.1", "Detailed System Hyperparameters, Neural Dimensions, and Training Configurations", "99")
    ]

    tbl = doc.add_table(rows=len(tbl_items) + 1, cols=3)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="2E5B82"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="2E5B82"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="none"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tbl._tbl.tblPr.append(borders)
    
    hdr = tbl.rows[0]
    hdr.cells[0].width = Cm(2.8)
    hdr.cells[1].width = Cm(9.7)
    hdr.cells[2].width = Cm(2.0)
    
    for c_i, h_txt in enumerate(["TABLE NO.", "TITLE / DESCRIPTION", "PAGE"]):
        cell = hdr.cells[c_i]
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="EBF2FA"/>')
        cell._tc.get_or_add_tcPr().append(shd)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        if c_i == 2:
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = p.add_run(h_txt)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
        r.bold = True

    for idx, (t_no, t_desc, t_pg) in enumerate(tbl_items):
        row = tbl.rows[idx + 1]
        row.cells[0].width = Cm(2.8)
        row.cells[1].width = Cm(9.7)
        row.cells[2].width = Cm(2.0)
        
        p0 = row.cells[0].paragraphs[0]
        p0.paragraph_format.space_before = Pt(3)
        p0.paragraph_format.space_after = Pt(3)
        r0 = p0.add_run(t_no)
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(10)
        r0.bold = True
        
        p1 = row.cells[1].paragraphs[0]
        p1.paragraph_format.space_before = Pt(3)
        p1.paragraph_format.space_after = Pt(3)
        r1 = p1.add_run(t_desc)
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(10)
        
        p2 = row.cells[2].paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p2.paragraph_format.space_before = Pt(3)
        p2.paragraph_format.space_after = Pt(3)
        r2 = p2.add_run(t_pg)
        r2.font.name = 'Times New Roman'
        r2.font.size = Pt(10)
        r2.bold = True

    doc.add_page_break()

def add_list_of_acronyms_page(doc):
    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_h.paragraph_format.space_before = Pt(18)
    p_h.paragraph_format.space_after = Pt(18)
    r_h = p_h.add_run("LIST OF ACRONYMS AND ABBREVIATIONS")
    r_h.font.name = 'Times New Roman'
    r_h.font.size = Pt(16)
    r_h.bold = True

    acronyms = [
        ("AI", "Artificial Intelligence"),
        ("AP", "Average Precision (Evaluation Metric for Information Retrieval & Detection)"),
        ("AUC", "Area Under the Receiver Operating Characteristic Curve"),
        ("BLAS", "Basic Linear Algebra Subprograms (High-Performance Vector Math)"),
        ("BPR", "Bayesian Personalized Ranking (Pairwise Ranking Loss Function)"),
        ("CE", "Cross-Entropy (Categorical Classification Loss)"),
        ("CF", "Collaborative Filtering (Classic Recommendation Paradigm)"),
        ("CUDA", "Compute Unified Device Architecture (NVIDIA GPU Computing API)"),
        ("CVR", "Conversion Rate (Simulated Commercial Purchase Ratio)"),
        ("CTR", "Click-Through Rate (Simulated Commercial Interaction Probability)"),
        ("GCL", "Graph Contrastive Learning (Self-Supervised Representation Alignment)"),
        ("GCN", "Graph Convolutional Network (Neighborhood Message-Passing Architecture)"),
        ("GKPI", "General Key Performance Indicator (Unified Harmonic Recommendation Metric)"),
        ("g-LTC", "Genuine Long-Tail Coverage (Catalog Coverage Excluding Attack Targets)"),
        ("GMV", "Gross Merchandise Value (Simulated E-Commerce Economic Metric)"),
        ("GNN", "Graph Neural Network"),
        ("InfoNCE", "Information Noise-Contrastive Estimation (Contrastive Loss Objective)"),
        ("KL", "Kullback-Leibler Divergence (Relative Entropy Between Distributions)"),
        ("LightGCN", "Lightweight Graph Convolutional Network (Linearized Message Passing)"),
        ("LLM", "Large Language Model (Generative Pretrained Transformer for Text Auditing)"),
        ("LTC", "Long-Tail Coverage (Raw Unique Tail Items Recommended Proportion)"),
        ("MIRF", "Multi-view Interaction Reliability Filter (Our Proposed Fusion Mechanism)"),
        ("MRMC", "Mean Rank Miscalibration (Popularity Distribution KL Divergence)"),
        ("nDCG", "Normalized Discounted Cumulative Gain (Graded Ranking Quality Metric)"),
        ("NeuMF", "Neural Matrix Factorization (Deep Collaborative Filtering Backbone)"),
        ("NLL", "Negative Log-Likelihood (Probabilistic Model Objective)"),
        ("PR", "Precision-Recall (Performance Curve for Imbalanced Classification)"),
        ("RMSE-PC", "Root Mean Square Error of Popularity Calibration"),
        ("ROC", "Receiver Operating Characteristic (True Positive Rate vs. False Positive Rate)"),
        ("RRFN", "Robust Risk-consistent Fusion Network (Stochastic Noise Transition Engine)"),
        ("SGL", "Self-supervised Graph Learning (Graph Contrastive Learning with Dropout)"),
        ("SimGCL", "Simple Graph Contrastive Learning (GCL with Latent Uniform Noise)"),
        ("STER", "Spam Target Exposure Rate (Fraction of Malicious Target Items Pushed)"),
        ("SUP / SUPER", "Soft Blueprint Merging for Calibrated Long-Tail Recommendation"),
        ("SVD", "Singular Value Decomposition (Matrix Spectral Factorization)"),
        ("T-CE", "Transition Matrix Cross-Entropy (Implicit Feedback Denoising Baseline)"),
        ("Trust-GRS", "Trustworthy Graph Recommendation System (Baseline Shilling Defense)"),
        ("VaeCF", "Variational Autoencoder for Collaborative Filtering"),
        ("WAL", "Write-Ahead Logging (High-Concurrency SQLite Database Journaling Mode)")
    ]

    tbl = doc.add_table(rows=len(acronyms) + 1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="2E5B82"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="2E5B82"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E1E4E8"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tbl._tbl.tblPr.append(borders)
    
    hdr = tbl.rows[0]
    hdr.cells[0].width = Cm(3.5)
    hdr.cells[1].width = Cm(11.0)
    
    for c_i, h_txt in enumerate(["ACRONYM", "EXPANDED MEANING / DEFINITION"]):
        cell = hdr.cells[c_i]
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="EBF2FA"/>')
        cell._tc.get_or_add_tcPr().append(shd)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(h_txt)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
        r.bold = True

    for idx, (acr, full_name) in enumerate(acronyms):
        row = tbl.rows[idx + 1]
        row.cells[0].width = Cm(3.5)
        row.cells[1].width = Cm(11.0)
        
        is_even = (idx % 2 == 1)
        fill = "F9FBFC" if is_even else "FFFFFF"
        
        for c_i, txt in enumerate([acr, full_name]):
            cell = row.cells[c_i]
            if is_even:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill}"/>')
                cell._tc.get_or_add_tcPr().append(shd)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2.5)
            p.paragraph_format.space_after = Pt(2.5)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(txt)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)
            if c_i == 0:
                r.bold = True

    doc.add_page_break()
