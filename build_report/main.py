"""
Main Orchestrator for Generating BCSE497J Comprehensive Individual Project Reports
Generates 3 individualized reports for all team members adhering strictly to
the VIT Chennai BCE497J Project Report Template Guidelines.
"""

import os
import shutil
import docx

from report_builder import (
    create_base_document, set_section_margins, set_section_page_numbering,
    add_footer_page_number
)
from section_frontmatter import (
    add_cover_page, add_title_page, add_declaration_page, add_certificate_page,
    add_abstract_page, add_acknowledgement_page,
    add_table_of_contents_page, add_list_of_figures_page, add_list_of_tables_page,
    add_list_of_acronyms_page
)
from chapter1 import build_chapter_1
from chapter2 import build_chapter_2
from chapter3 import build_chapter_3
from chapter4 import build_chapter_4
from chapter5 import build_chapter_5
from chapter6 import build_chapter_6
from section_backmatter import build_references, build_appendices
from student_configs import STUDENT_PROFILES

def generate_single_report(student_config, include_inner_title_page=True):
    cand_name = student_config["name"]
    cand_reg = student_config["reg_no"]
    file_prefix = student_config["file_prefix"]

    print("=" * 75)
    print(f"GENERATING REPORT FOR: {cand_name} ({cand_reg})")
    print(f"PILLAR: {student_config['pillar_title']}")
    print("=" * 75)

    # 1. Base Document & Section 1: Cover and Title Page (No page numbers)
    doc = create_base_document()
    sec1 = doc.sections[0]
    set_section_margins(sec1)
    
    print(f"-> Adding Cover Page (Page 1) with candidate {cand_name} first...")
    add_cover_page(doc, student_config=student_config)
    
    if include_inner_title_page:
        print(f"-> Adding Title Page (Page 2) with candidate {cand_name} first...")
        add_title_page(doc, student_config=student_config)

    # 2. Section 2: Preliminary Pages / Front Matter (Lower Roman: i, ii, iii...)
    print("-> Configuring Section 2 for Front Matter (Roman page numbers)...")
    sec2 = doc.add_section()
    set_section_margins(sec2)
    set_section_page_numbering(sec2, fmt='lowerRoman', start=3) # Page iii
    add_footer_page_number(sec2)

    print(f"-> Adding Declaration (Individual Candidate: {cand_name})...")
    add_declaration_page(doc, student_config=student_config)

    print(f"-> Adding Certificate Page (Individual Candidate: {cand_name})...")
    add_certificate_page(doc, student_config=student_config)

    # Note: External Guide Certificate is skipped as it is not applicable for in-house research projects

    print("-> Adding Abstract Page (Strictly 1 Page, Unified)...")
    add_abstract_page(doc, student_config=student_config)

    print(f"-> Adding Acknowledgement Page (Individual Candidate: {cand_name})...")
    add_acknowledgement_page(doc, student_config=student_config)

    print("-> Adding Table of Contents (CONTENTS)...")
    add_table_of_contents_page(doc, student_config=student_config)

    print("-> Adding List of Figures...")
    add_list_of_figures_page(doc)

    print("-> Adding List of Tables (including Table 1.1 Work Division)...")
    add_list_of_tables_page(doc)

    print("-> Adding List of Acronyms and Abbreviations...")
    add_list_of_acronyms_page(doc)

    # 3. Section 3: Main Body Chapters (Arabic Numerals: 1, 2, 3...)
    print("-> Configuring Section 3 for Main Chapters (Arabic page numbers starting at 1)...")
    sec3 = doc.add_section()
    set_section_margins(sec3)
    set_section_page_numbering(sec3, fmt='decimal', start=1)
    add_footer_page_number(sec3)

    print(f"-> Building Chapter 1 with Work Division Table 1.1 and {cand_name} technical contribution focus...")
    build_chapter_1(doc, student_config=student_config)

    print("-> Building Chapter 2: Literature Review and Theoretical Background...")
    build_chapter_2(doc)

    print("-> Building Chapter 3: System Design & Mathematical Methodology...")
    build_chapter_3(doc, student_config=student_config)

    print("-> Building Chapter 4: System Implementation & Architecture Details...")
    build_chapter_4(doc, student_config=student_config)

    print("-> Building Chapter 5: Experimental Evaluation, Results, and Discussion...")
    build_chapter_5(doc, student_config=student_config)

    print("-> Building Chapter 6: Conclusion and Future Work...")
    build_chapter_6(doc, student_config=student_config)

    print("-> Building References (Strict APA Format)...")
    build_references(doc)

    print(f"-> Building Appendices with Plagiarism Statement for {cand_name} ({cand_reg})...")
    build_appendices(doc, student_config=student_config)

    # 4. Save documents
    out_dir = r"paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path_paper = os.path.join(out_dir, f"{file_prefix}.docx")
    out_path_root = f"{file_prefix}.docx"

    print(f"-> Saving document to {out_path_paper}...")
    doc.save(out_path_paper)

    print(f"-> Copying document to {out_path_root}...")
    try:
        shutil.copyfile(out_path_paper, out_path_root)
        print(f"-> Successfully updated {out_path_root}")
    except PermissionError:
        print(f"-> Notice: {out_path_root} is currently open in Microsoft Word. Saved to {out_path_paper} successfully.")

    # Also update generic project_report.docx for Advait
    if student_config["id"] == "advait":
        generic_paper = os.path.join(out_dir, "project_report.docx")
        generic_root = "project_report.docx"
        try:
            shutil.copyfile(out_path_paper, generic_paper)
            shutil.copyfile(out_path_paper, generic_root)
            print(f"-> Also updated {generic_paper} and {generic_root}")
        except Exception as e:
            print(f"-> Notice for generic copy: {e}")

    print(f"COMPLETED REPORT FOR {cand_name}!\n")

def generate_all_reports():
    print("=" * 80)
    print("BCSE497J MULTI-STUDENT COMPREHENSIVE PROJECT REPORT GENERATION")
    print("=" * 80)
    for student in STUDENT_PROFILES:
        generate_single_report(student, include_inner_title_page=True)
    print("=" * 80)
    print("ALL 3 INDIVIDUAL PROJECT REPORTS SUCCESSFULLY GENERATED!")
    print("=" * 80)

if __name__ == "__main__":
    generate_all_reports()
