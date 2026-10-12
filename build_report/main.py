"""
Main Orchestrator for Generating BCSE497J Comprehensive Project Report
Follows VIT Chennai BCE497J Project Report Template Guidelines.
"""

import os
import shutil
import docx

from report_builder import (
    create_base_document, set_section_margins, set_section_page_numbering,
    add_footer_page_number
)
from section_frontmatter import (
    add_cover_page, add_declaration_page, add_certificate_page,
    add_external_certificate_page, add_abstract_page, add_acknowledgement_page,
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

def generate_report():
    print("=" * 70)
    print("STARTING BCSE497J COMPREHENSIVE PROJECT REPORT GENERATION")
    print("=" * 70)

    # 1. Base Document & Section 1: Cover and Title Page (No page numbers)
    doc = create_base_document()
    sec1 = doc.sections[0]
    set_section_margins(sec1)
    
    print("-> Generating Cover Page (Page 1) with all team members...")
    add_cover_page(doc, is_title_page=False)
    
    print("-> Generating Title Page (Page 2) with all team members...")
    add_cover_page(doc, is_title_page=True)

    # 2. Section 2: Preliminary Pages / Front Matter (Lower Roman: i, ii, iii...)
    print("-> Configuring Section 2 for Front Matter (Roman page numbers)...")
    sec2 = doc.add_section()
    set_section_margins(sec2)
    set_section_page_numbering(sec2, fmt='lowerRoman', start=3) # Page iii
    add_footer_page_number(sec2)

    print("-> Adding Declaration (Candidate: Advait Ravindrakumar Varhade)...")
    add_declaration_page(doc)

    print("-> Adding Certificate Page (Guide: Dr. XYZ, HoD: HOD NAME)...")
    add_certificate_page(doc)

    print("-> Adding External Guide Certificate (Not Applicable)...")
    add_external_certificate_page(doc)

    print("-> Adding Abstract Page...")
    add_abstract_page(doc)

    print("-> Adding Acknowledgement Page...")
    add_acknowledgement_page(doc)

    print("-> Adding Table of Contents (CONTENTS)...")
    add_table_of_contents_page(doc)

    print("-> Adding List of Figures...")
    add_list_of_figures_page(doc)

    print("-> Adding List of Tables...")
    add_list_of_tables_page(doc)

    print("-> Adding List of Acronyms and Abbreviations...")
    add_list_of_acronyms_page(doc)

    # 3. Section 3: Main Body Chapters (Arabic Numerals: 1, 2, 3...)
    print("-> Configuring Section 3 for Main Chapters (Arabic page numbers starting at 1)...")
    sec3 = doc.add_section()
    set_section_margins(sec3)
    set_section_page_numbering(sec3, fmt='decimal', start=1)
    add_footer_page_number(sec3)

    print("-> Building Chapter 1: Introduction and Motivation...")
    build_chapter_1(doc)

    print("-> Building Chapter 2: Literature Review and Theoretical Background...")
    build_chapter_2(doc)

    print("-> Building Chapter 3: System Design and Mathematical Methodology...")
    build_chapter_3(doc)

    print("-> Building Chapter 4: System Implementation and Architecture Details...")
    build_chapter_4(doc)

    print("-> Building Chapter 5: Experimental Evaluation, Results, and Discussion...")
    build_chapter_5(doc)

    print("-> Building Chapter 6: Conclusion and Future Work...")
    build_chapter_6(doc)

    print("-> Building References (Strict APA Format)...")
    build_references(doc)

    print("-> Building Appendices (Source Tree, Hyperparameters, Code Listings, Plagiarism Compliance)...")
    build_appendices(doc)

    # 4. Save documents
    out_dir = r"paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path_paper = os.path.join(out_dir, "project_report.docx")
    out_path_root = "project_report.docx"

    print(f"-> Saving document to {out_path_paper}...")
    doc.save(out_path_paper)

    print(f"-> Copying document to {out_path_root}...")
    shutil.copyfile(out_path_paper, out_path_root)

    print("=" * 70)
    print("PROJECT REPORT GENERATION COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    generate_report()
