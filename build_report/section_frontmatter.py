"""
Front Matter Generator for BCSE497J Project Report
Contains Cover Page, Title Page, Declaration, Certificate, Abstract,
Acknowledgement, Table of Contents, List of Figures, List of Tables, and List of Acronyms.
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

def add_cover_page(doc, is_title_page=False):
    # Top spacing
    p_top = doc.add_paragraph()
    p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_top.paragraph_format.space_before = Pt(12)
    p_top.paragraph_format.space_after = Pt(4)
    r_top = p_top.add_run("A project report on")
    r_top.font.name = 'Times New Roman'
    r_top.font.size = Pt(12)
    r_top.italic = True

    # Title of Project
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(8)
    p_title.paragraph_format.space_after = Pt(14)
    p_title.paragraph_format.line_spacing = 1.15
    r_title = p_title.add_run("ROBUST RISK-CONSISTENT GRAPH-CONTRASTIVE LEARNING FOR POPULARITY-CALIBRATED RECOMMENDATION WITH MIRF AUDITING")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(18)
    r_title.bold = True

    # Submitted in partial fulfillment...
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(6)
    p_sub.paragraph_format.space_after = Pt(4)
    r_sub = p_sub.add_run("Submitted in partial fulfillment for the award of the degree of")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(13)
    r_sub.italic = True

    # Degree
    p_deg = doc.add_paragraph()
    p_deg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_deg.paragraph_format.space_before = Pt(4)
    p_deg.paragraph_format.space_after = Pt(12)
    r_deg = p_deg.add_run("Bachelor of Technology in Computer Science and Engineering")
    r_deg.font.name = 'Times New Roman'
    r_deg.font.size = Pt(16)
    r_deg.bold = True

    # by
    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_by.paragraph_format.space_before = Pt(4)
    p_by.paragraph_format.space_after = Pt(6)
    r_by = p_by.add_run("by")
    r_by.font.name = 'Times New Roman'
    r_by.font.size = Pt(13)
    r_by.italic = True

    # Candidate Names & Reg Nos (Group project: 1st & 2nd page list all team members)
    p_cand = doc.add_paragraph()
    p_cand.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cand.paragraph_format.space_before = Pt(4)
    p_cand.paragraph_format.space_after = Pt(16)
    p_cand.paragraph_format.line_spacing = 1.25

    names = [
        ("ADVAIT RAVINDRAKUMAR VARHADE", "23BCE5098"),
        ("P. MANAS", "23BCE5076"),
        ("VIVAASH KOUNDAL", "23BCE5095")
    ]
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
    r_dt = p_dt.add_run("October, 2026")
    r_dt.font.name = 'Times New Roman'
    r_dt.font.size = Pt(12)

    doc.add_page_break()

def add_declaration_page(doc):
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
        "I hereby declare that the thesis entitled “ROBUST RISK-CONSISTENT GRAPH-CONTRASTIVE LEARNING FOR POPULARITY-CALIBRATED RECOMMENDATION WITH MIRF AUDITING” "
        "submitted by ADVAIT RAVINDRAKUMAR VARHADE (23BCE5098), for the award of the degree of Bachelor of Technology in Computer Science and Engineering, "
        "Vellore Institute of Technology, Chennai is a record of bonafide work carried out by me under the supervision of [Dr. XYZ, Associate Professor], "
        "School of Computer Science and Engineering, Vellore Institute of Technology, Chennai."
    )

    add_paragraph(
        doc,
        "I further declare that the work reported in this thesis has not been submitted and will not be submitted, either in part or in full, "
        "for the award of any other degree or diploma in this institute or any other institute or university."
    )

    # Place, Date, Signature
    p_sign = doc.add_paragraph()
    p_sign.paragraph_format.space_before = Pt(40)
    p_sign.paragraph_format.space_after = Pt(6)
    p_sign.paragraph_format.line_spacing = 1.3
    
    r_pl = p_sign.add_run("Place: Chennai\nDate: October 2026")
    r_pl.font.name = 'Times New Roman'
    r_pl.font.size = Pt(12)
    r_pl.bold = True
    
    # Signature on right
    p_cand = doc.add_paragraph()
    p_cand.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_cand.paragraph_format.space_before = Pt(10)
    p_cand.paragraph_format.space_after = Pt(20)
    r_cand = p_cand.add_run("Signature of the Candidate\nADVAIT RAVINDRAKUMAR VARHADE\n(Reg. No. 23BCE5098)")
    r_cand.font.name = 'Times New Roman'
    r_cand.font.size = Pt(12)
    r_cand.bold = True

    add_callout_box(
        doc,
        "SUBMISSION NOTE: As per the BCSE497J project evaluation instructions, after the final offline panel review, "
        "a high-resolution scanned photograph of this physically signed and endorsed Declaration page will be affixed "
        "directly into this soft copy document prior to final submission on VTOP.",
        title="POST-REVIEW SUBMISSION COMPLIANCE"
    )

    doc.add_page_break()

def add_certificate_page(doc):
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
        "This is to certify that the report entitled “ROBUST RISK-CONSISTENT GRAPH-CONTRASTIVE LEARNING FOR POPULARITY-CALIBRATED RECOMMENDATION WITH MIRF AUDITING” "
        "is prepared and submitted by ADVAIT RAVINDRAKUMAR VARHADE (23BCE5098) to Vellore Institute of Technology, Chennai, "
        "in partial fulfillment of the requirement for the award of the degree of Bachelor of Technology in Computer Science and Engineering "
        "is a bonafide record of work carried out under my guidance. The project fulfills the requirements as per the regulations of this University "
        "and in my opinion meets the necessary academic and technical standards for submission. The contents of this report have not been submitted "
        "and will not be submitted either in part or in full, for the award of any other degree or diploma and the same is certified."
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
    r00 = p00.add_run("Signature of the Project Guide:\nName: [Dr. XYZ, Associate Professor]\nDate: October 2026")
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
    r20 = p20.add_run("Approved by the Head of Department:\nDepartment of Computer Science and Engineering\nName: [HOD NAME]\nDate:")
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

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(12)

    add_callout_box(
        doc,
        "OFFICIAL ENDORSEMENT NOTICE: Following the completion of the Pre-Final and Final Offline Viva-Voce Review Panel, "
        "the physical signatures of the Project Guide, Panel Examiners, and Head of Department along with the official School Seal "
        "will be stamped on this Certificate page. A clear photograph/scan will be integrated prior to VTOP submission.",
        title="CERTIFICATE COMPLIANCE"
    )

    doc.add_page_break()

def add_external_certificate_page(doc):
    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_h.paragraph_format.space_before = Pt(18)
    p_h.paragraph_format.space_after = Pt(24)
    r_h = p_h.add_run("CERTIFICATE BY EXTERNAL GUIDE")
    r_h.font.name = 'Times New Roman'
    r_h.font.size = Pt(14)
    r_h.bold = True
    r_h.underline = True

    add_paragraph(
        doc,
        "NOT APPLICABLE"
    )
    add_paragraph(
        doc,
        "This project entitled “ROBUST RISK-CONSISTENT GRAPH-CONTRASTIVE LEARNING FOR POPULARITY-CALIBRATED RECOMMENDATION WITH MIRF AUDITING” "
        "has been carried out purely as an in-house internal academic and research project within the School of Computer Science and Engineering (SCOPE), "
        "Vellore Institute of Technology, Chennai. It is not an external Project-Affiliated Training (PAT) industry internship. "
        "Hence, external industry guide endorsement is not applicable."
    )

    doc.add_page_break()

def add_abstract_page(doc):
    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_h.paragraph_format.space_before = Pt(18)
    p_h.paragraph_format.space_after = Pt(18)
    r_h = p_h.add_run("ABSTRACT")
    r_h.font.name = 'Times New Roman'
    r_h.font.size = Pt(14)
    r_h.bold = True
    r_h.underline = True

    add_paragraph(
        doc,
        "Modern web-scale recommender systems operate on bipartite user-item interaction graphs that suffer from two co-occurring "
        "systemic vulnerabilities: (1) severe susceptibility to coordinated adversarial shilling attacks and temporal review-bombing bursts "
        "that inject deceptive feedback to distort item exposure, and (2) extreme catalog popularity bias where Pareto-skewed interaction "
        "distributions starve genuine long-tail items of visibility. Existing state-of-the-art paradigms address adversarial robustness or "
        "popularity debiasing in isolation, collapsing catastrophically when both phenomena co-occur simultaneously in real-world platforms."
    )

    add_paragraph(
        doc,
        "In this project report, we design, implement, mathematically validate, and empirically evaluate RRFN-LLM-SUPER, a novel end-to-end "
        "framework uniting stochastic noise-transition estimation, the Multi-view Interaction Reliability Filter (MIRF), risk-aware Graph "
        "Contrastive Learning (GCL), and calibrated Pareto debiasing. We establish formal mathematical foundations, proving that Tikhonov-regularized "
        "noise inversion strictly upper-bounds operator norms by ||T_lambda^dagger||_2 <= 1 / (2 * sqrt(lambda)), guaranteeing non-singular gradient "
        "stability without numerical divergence, and establish asymptotic surrogate risk consistency O(delta + 1/sqrt(N)) alongside confidence "
        "inversion resolvent bounds O(delta/sqrt(lambda) + sqrt(lambda))."
    )

    add_paragraph(
        doc,
        "The proposed MIRF ensemble computes continuous sample reliability weights w_ui in [0, 1] by synthesizing statistical model prediction consistency, "
        "asynchronous Large Language Model (LLM) semantic review verification with thread-safe SQLite WAL caching, and 2D BLAS vectorized burst scoring "
        "(achieving a measured 4.95 ms latency for 42,000 interactions, representing a >500x speedup over standard nested iteration). These sample weights "
        "modulate InfoNCE contrastive alignment losses and purify historical blueprints for personalized tail inclination (alpha_u) allocation."
    )

    add_paragraph(
        doc,
        "Extensive empirical evaluations across 5 independent random seeds (42–46) on three multi-modal benchmark domains (MovieLens-1M sub-graph, "
        "Amazon Reviews with SimGCL, and Yelp Academic with SGL) demonstrate that RRFN-LLM-SUPER achieves an Attack Detection Average Precision (AP) "
        "of 0.6884 on Amazon and 0.6816 on Yelp (a 3.9–4.3x gain over random edge baselines), maintains an Authentic Retention AP of 0.9989 (ROC-AUC = 0.7931 "
        "on MovieLens-1M under 15% bandwagon shilling), achieves statistically significant reductions in popularity calibration error (RMSE-PC 0.1067, p = 0.0166; "
        "MRMC 0.1128, p = 0.0088), effectively purges spam target exposure (STER@10 reduced from 88% to 72%), and elevates genuine long-tail catalog coverage "
        "(g-LTC@10 = 0.4202 vs. 0.3371, p = 0.0397 on Yelp). Finally, an offline 1,000-sample Monte Carlo economic simulation demonstrates statistically significant "
        "resilience against revenue distortion (p < 0.001)."
    )

    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_before = Pt(12)
    p_kw.paragraph_format.space_after = Pt(6)
    r_kwh = p_kw.add_run("Keywords: ")
    r_kwh.font.name = 'Times New Roman'
    r_kwh.font.size = Pt(11)
    r_kwh.bold = True
    r_kwt = p_kw.add_run("Recommender Systems, Graph Neural Networks, Graph Contrastive Learning, Multi-view Interaction Reliability Filter (MIRF), "
                         "Review-Bombing Detection, Popularity Bias, Long-Tail Calibration, Large Language Models, Trustworthy AI, Risk Consistency.")
    r_kwt.font.name = 'Times New Roman'
    r_kwt.font.size = Pt(11)
    r_kwt.italic = True

    doc.add_page_break()

def add_acknowledgement_page(doc):
    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_h.paragraph_format.space_before = Pt(18)
    p_h.paragraph_format.space_after = Pt(18)
    r_h = p_h.add_run("ACKNOWLEDGEMENT")
    r_h.font.name = 'Times New Roman'
    r_h.font.size = Pt(14)
    r_h.bold = True

    add_paragraph(
        doc,
        "It is my utmost pleasure to express with a deep sense of gratitude to [Dr. XYZ, Associate Professor], "
        "School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, for his/her constant guidance, continual encouragement, "
        "understanding, and intellectual mentorship throughout this endeavor. His/her deep insights and rigorous critique in the domains of "
        "Recommender Systems, Graph Machine Learning, and Trustworthy AI have been invaluable in shaping the theoretical and experimental dimensions of this research."
    )

    add_paragraph(
        doc,
        "It is with profound gratitude that I extend my heartfelt thanks to the visionary leadership of Vellore Institute of Technology: "
        "Dr. G. Viswanathan, our Honorable Chancellor; Mr. Sankar Viswanathan, Dr. Sekar Viswanathan, and Dr. G. V. Selvam, Vice Presidents; "
        "Dr. Sandhya Pentareddy, Executive Director; Ms. Kadhambari S. Viswanathan, Assistant Vice-President; "
        "Dr. V. S. Kanchana Bhaaskaran, Vice-Chancellor; Dr. T. Thyagarajan, Pro-Vice Chancellor, VIT Chennai; and Dr. P. K. Manoharan, Additional Registrar, "
        "for providing an exceptional academic environment, cutting-edge computing infrastructure, and continual inspiration during the course of my study."
    )

    add_paragraph(
        doc,
        "Special mention and sincere thanks are extended to Dr. Viswanathan V, Dean; Dr. Nithyanandam P, Associate Dean; "
        "Dr. Suganya G, Associate Dean; and Dr. Sweetlin Hemalatha C, Associate Dean, School of Computer Science and Engineering, "
        "Vellore Institute of Technology, Chennai, for spending their valuable time and efforts in sharing their academic knowledge and supporting us in every phase."
    )

    add_paragraph(
        doc,
        "In a jubilant state, I express my whole-hearted thanks to [HOD NAME], Head of the Department, B.Tech. Computer Science and Engineering, "
        "as well as the BCSE497J Project Coordinators and faculty committee for their structured academic framework, periodic reviews, and constant encouragement."
    )

    add_paragraph(
        doc,
        "I also thank my project teammates, P. Manas (23BCE5076) and Vivaash Koundal (23BCE5095), for their collaborative discussions and camaraderie. "
        "My deepest and most affectionate gratitude goes to my parents and family members for their unconditional sacrifices, moral backing, and prayers. "
        "Finally, I thank all my friends and peers at VIT Chennai who supported me throughout this journey."
    )

    p_sign = doc.add_paragraph()
    p_sign.paragraph_format.space_before = Pt(28)
    p_sign.paragraph_format.space_after = Pt(6)
    p_sign.paragraph_format.line_spacing = 1.3
    
    r_pl = p_sign.add_run("Place: Chennai\nDate: October 2026")
    r_pl.font.name = 'Times New Roman'
    r_pl.font.size = Pt(12)
    r_pl.bold = True
    
    p_cand = doc.add_paragraph()
    p_cand.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_cand.paragraph_format.space_before = Pt(8)
    p_cand.paragraph_format.space_after = Pt(12)
    r_cand = p_cand.add_run("ADVAIT RAVINDRAKUMAR VARHADE\n(Reg. No. 23BCE5098)")
    r_cand.font.name = 'Times New Roman'
    r_cand.font.size = Pt(12)
    r_cand.bold = True

    doc.add_page_break()

def add_table_of_contents_page(doc):
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
        ("ABSTRACT", "vi"),
        ("ACKNOWLEDGEMENT", "vii"),
        ("CONTENTS", "viii"),
        ("LIST OF FIGURES", "x"),
        ("LIST OF TABLES", "xii"),
        ("LIST OF ACRONYMS", "xiii"),
        ("CHAPTER 1   INTRODUCTION", "1"),
        ("   1.1  Background and Motivation", "1"),
        ("   1.2  Systemic Vulnerabilities in Graph Recommenders", "3"),
        ("   1.3  Problem Statement and Formal Challenges", "5"),
        ("   1.4  Objectives of the Project", "6"),
        ("   1.5  Grounding Key Concepts: SUP, RRFN, and MIRF", "7"),
        ("   1.6  Scope and Boundaries of the Work", "9"),
        ("   1.7  Organization of the Project Report", "10"),
        ("CHAPTER 2   LITERATURE REVIEW AND THEORETICAL BACKGROUND", "12"),
        ("   2.1  Overview of Collaborative Filtering Paradigms", "12"),
        ("   2.2  Adversarial Shilling Attacks in Recommender Systems", "14"),
        ("   2.3  Noise-Transition Modeling and Loss Correction", "16"),
        ("   2.4  Catalog Popularity Skew and Long-Tail Calibration", "18"),
        ("   2.5  Soft Blueprint Merging (SUPER/SUP) and Vulnerabilities", "20"),
        ("   2.6  Graph Contrastive Learning (GCL: SGL, SimGCL)", "22"),
        ("   2.7  Trustworthy Graph Learning & Positioning vs. Trust-GRS", "24"),
        ("   2.8  Large Language Models (LLMs) in Recommendation Auditing", "26"),
        ("   2.9  Critical Research Gaps in Prior Literature", "27"),
        ("   2.10 Comprehensive Literature Positioning Matrix (Table 2.1)", "28"),
        ("CHAPTER 3   SYSTEM DESIGN AND MATHEMATICAL METHODOLOGY", "30"),
        ("   3.1  End-to-End System Architecture (Figure 3.1)", "30"),
        ("   3.2  Mathematical Formulations and Noise Transition Model", "32"),
        ("   3.3  Stage 1: Anchor Selection & Tikhonov Noise Inversion", "34"),
        ("   3.4  Stage 2: Multi-View Interaction Reliability Filter (MIRF)", "36"),
        ("       3.4.1 Statistical Prediction Consistency (R_RRFN)", "36"),
        ("       3.4.2 LLM Review Auditing with SQLite WAL Cache (R_LLM)", "37"),
        ("       3.4.3 2D BLAS Vectorized Review-Bombing Burst Detection", "38"),
        ("       3.4.4 Continuous Signal Synthesis and Trust Weights (w_ui)", "40"),
        ("   3.5  Stage 3: Risk-Aware Graph Contrastive Learning", "41"),
        ("       3.5.1 SimGCL Latent Perturbations & SGL Structural Dropout", "41"),
        ("       3.5.2 Sample-Weighted InfoNCE Contrastive Objective", "42"),
        ("       3.5.3 Dual Partition Training (Head and Tail Models)", "43"),
        ("   3.6  Stage 4: Denoised SUPER Blueprint & Trajectory Interleaving", "44"),
        ("   3.7  Theoretical Proofs and Formal Mathematical Guarantees", "46"),
        ("       3.7.1 Theorem 1: Spectral Operator Norm Bound Proof", "46"),
        ("       3.7.2 Theorem 2: Surrogate Excess Risk Consistency Proof", "48"),
        ("       3.7.3 Confidence Inversion Resolvent Error Bounds", "50"),
        ("   3.8  Formal Definitions of Recommendation and Detection Metrics", "51"),
        ("CHAPTER 4   SYSTEM IMPLEMENTATION AND ARCHITECTURE DETAILS", "53"),
        ("   4.1  Software Architecture and Design Principles", "53"),
        ("   4.2  Comprehensive Module Mapping and File Index (Table 4.1)", "54"),
        ("   4.3  Multi-Domain Data Ingestion and Robust Loading Pipelines", "56"),
        ("   4.4  Graph Neural Network Backbones and Contrastive Engine", "58"),
        ("   4.5  MIRF Filter Modules and High-Performance BLAS Acceleration", "60"),
        ("   4.6  Denoised SUPER Calibrated Recommender Engine", "62"),
        ("   4.7  Algorithmic Workflows and End-to-End Pseudocode", "64"),
        ("   4.8  Interactive Telemetry and Streamlit Web Dashboard", "66"),
        ("   4.9  Software Hardening, Numerical Protection, and Concurrency", "68"),
        ("CHAPTER 5   EXPERIMENTAL EVALUATION, RESULTS, AND DISCUSSION", "70"),
        ("   5.1  Experimental Setup and Dataset Characteristics (Table 5.1)", "70"),
        ("   5.2  Adversarial Attack Simulation Protocols", "72"),
        ("   5.3  Multi-Seed Statistical Validation Protocol (Seeds 42-46)", "74"),
        ("   5.4  Empirical Benchmark Results on MovieLens-1M (Table 5.2)", "75"),
        ("   5.5  Multi-Domain Benchmarking on Amazon and Yelp (Table 5.3)", "77"),
        ("   5.6  MIRF Attack Detection vs. Authentic Retention Analysis", "79"),
        ("   5.7  Component Ablation Study and Hyperparameter Sweep (Table 5.4)", "81"),
        ("   5.8  Dual Partition Model Convergence and Training Loss Curves", "83"),
        ("   5.9  Scenario-Based Offline Economic Sensitivity Simulation", "84"),
        ("   5.10 Discussion of Engineering Trade-Offs and System Limitations", "86"),
        ("CHAPTER 6   CONCLUSION AND FUTURE WORK", "88"),
        ("   6.1  Summary of Completed Work and Contributions", "88"),
        ("   6.2  Key Academic and Practical Findings", "89"),
        ("   6.3  Realistic System Limitations", "90"),
        ("   6.4  Future Research Avenues and Extensions", "91"),
        ("REFERENCES (APA FORMAT)", "93"),
        ("APPENDIX A   REPOSITORY ARCHITECTURE AND COMPLETE SOURCE TREE", "96"),
        ("APPENDIX B   DETAILED HYPERPARAMETER CONFIGURATIONS (Table B.1)", "98"),
        ("APPENDIX C   SELECTED CORE ALGORITHMIC IMPLEMENTATION LISTINGS", "99"),
        ("APPENDIX D   PLAGIARISM CHECK COMPLIANCE AND SIMILARITY REPORT", "101")
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
        ("Figure 3.1", "End-to-End System Architecture of the RRFN-LLM-SUPER Framework", "31"),
        ("Figure 3.2", "Detailed Multi-View Interaction Reliability Filter (MIRF) Tri-Signal Pipeline", "37"),
        ("Figure 3.3", "Denoised SUPER Blueprint Assembly and Calibrated Trajectory Interleaving Flow", "45"),
        ("Figure 4.1", "Interactive Streamlit Web Dashboard: 4-Panel Live Telemetry & Inspector", "67"),
        ("Figure 5.1", "Empirical ROC Curve and Precision-Recall Curves for Attack & Authentic Detection", "79"),
        ("Figure 5.2", "Sensitivity Sweep of MIRF Anomaly Weights Along the Simplices omega_1 + omega_2 = 1.0", "82"),
        ("Figure 5.3", "Dual Partition Head/Tail Training Convergence and InfoNCE Contrastive Loss Curves", "83"),
        ("Figure 5.4", "Multi-Domain Radar Comparison Across MovieLens-1M, Amazon Reviews, and Yelp Academic", "78")
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
        ("Table 2.1", "Comprehensive Literature Positioning Matrix: Adversarial Robustness vs. Debiasing", "29"),
        ("Table 3.1", "Detailed 14-Dimension Architectural Comparison (Standalone SUP vs. RRFN-LLM-SUPER)", "33"),
        ("Table 4.1", "Complete Software Architecture, Module Responsibility, and File System Index", "55"),
        ("Table 5.1", "Dataset Characteristics, Graph Topology, and Bandwagon Attack Injection Statistics", "71"),
        ("Table 5.2", "Empirical 5-Seed Benchmark Results on MovieLens-1M Sub-Graph under 15% Attack", "76"),
        ("Table 5.3", "Multi-Domain 5-Seed Benchmark on Amazon Reviews (SimGCL) and Yelp Academic (SGL)", "77"),
        ("Table 5.4", "Comprehensive Component Ablation Study on Amazon Reviews (SimGCL) under Attack", "81"),
        ("Table B.1", "Detailed System Hyperparameters, Neural Dimensions, and Training Configurations", "98")
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
