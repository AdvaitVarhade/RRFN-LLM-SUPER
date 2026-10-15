"""
Core Document Builder and Styling Utilities for BCSE497J Project Report
Configured according to VIT Chennai BCE497J Report Template Guidelines.
"""

import os
import re
import docx
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def create_base_document():
    doc = docx.Document()
    
    # Configure base section margins (A4 with 3.81cm left, 2.54cm right/top/bottom)
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(3.81)
    sec.right_margin = Cm(2.54)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    
    # Configure default Normal style
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    font.color.rgb = RGBColor(0, 0, 0)
    
    # Ensure XML elements for fonts explicitly specify Times New Roman for complex scripts
    rPr = style._element.get_or_add_rPr()
    rFonts = parse_xml(f'<w:rFonts {nsdecls("w")} w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>')
    rPr.append(rFonts)
    
    return doc

def set_section_margins(section):
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(3.81)
    section.right_margin = Cm(2.54)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)

def set_section_page_numbering(section, fmt='decimal', start=1):
    sectPr = section._sectPr
    pgNumType = sectPr.find(qn('w:pgNumType'))
    if pgNumType is None:
        pgNumType = OxmlElement('w:pgNumType')
        sectPr.append(pgNumType)
    pgNumType.set(qn('w:fmt'), fmt)
    if start is not None:
        pgNumType.set(qn('w:start'), str(start))

def add_footer_page_number(section):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    
    run = p.add_run()
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'PAGE'
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)

def add_page_break(doc):
    doc.add_page_break()

def add_chapter_heading(doc, chapter_num_str, title_str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.2
    
    run1 = p.add_run(chapter_num_str.upper() + '\n')
    run1.font.name = 'Times New Roman'
    run1.font.size = Pt(16)
    run1.bold = True
    
    run2 = p.add_run(title_str.upper())
    run2.font.name = 'Times New Roman'
    run2.font.size = Pt(16)
    run2.bold = True
    
    # Add a thin decorative rule below chapter title
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_before = Pt(4)
    p_space.paragraph_format.space_after = Pt(12)
    p_space.paragraph_format.line_spacing = 1.0

def add_section_heading(doc, heading_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.keep_with_next = True
    
    run = p.add_run(heading_text.upper())
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.bold = True

def add_subsection_heading(doc, heading_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.keep_with_next = True
    
    run = p.add_run(heading_text.upper())
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.bold = True

def add_subsubsection_heading(doc, heading_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.keep_with_next = True
    
    run = p.add_run(heading_text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.bold = True

def _append_text_with_math(p, text, font_size=Pt(12)):
    if not text:
        return
    # Preserve escaped literal dollars
    text_clean = text.replace(r'\$', '\uE000')
    if '$' in text_clean:
        parts = re.split(r'(\$[^\$]+\$)', text_clean)
        for part in parts:
            if part.startswith('$') and part.endswith('$') and len(part) > 2:
                latex = part[1:-1].replace('\uE000', '$')
                omml = latex_to_omml_element(latex)
                if omml is not None:
                    p._p.append(omml)
                else:
                    r = p.add_run(latex)
                    r.font.name = 'Cambria Math'
                    r.font.size = font_size
                    r.italic = True
            elif part:
                clean_part = part.replace('\uE000', '$')
                r = p.add_run(clean_part)
                r.font.name = 'Times New Roman'
                r.font.size = font_size
    else:
        clean_text = text_clean.replace('\uE000', '$')
        r = p.add_run(clean_text)
        r.font.name = 'Times New Roman'
        r.font.size = font_size

def add_paragraph(doc, text, bold_prefix=None, italic_prefix=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(12)
        r_pre.bold = True
    
    if italic_prefix:
        r_pre = p.add_run(italic_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(12)
        r_pre.italic = True
        
    _append_text_with_math(p, text, Pt(12))
    return p

def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(12)
        r_pre.bold = True
        
    _append_text_with_math(p, text, Pt(12))
    return p

def add_numbered(doc, num_str, text, bold_prefix=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.25)
    
    r_num = p.add_run(num_str + ' ')
    r_num.font.name = 'Times New Roman'
    r_num.font.size = Pt(12)
    r_num.bold = True
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(12)
        r_pre.bold = True
        
    _append_text_with_math(p, text, Pt(12))
    return p

_OMML_TRANSFORM = None

def _get_omml_transform():
    global _OMML_TRANSFORM
    if _OMML_TRANSFORM is not None:
        return _OMML_TRANSFORM
    
    import lxml.etree as ET
    
    candidates = [
        os.path.join(os.path.dirname(__file__), 'MML2OMML.XSL'),
        r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL',
        r'C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL',
        r'C:\Program Files\Microsoft Office\Office16\MML2OMML.XSL',
        r'C:\Program Files\Microsoft Office\Office15\MML2OMML.XSL',
        r'C:\Program Files\Microsoft Office\Office14\MML2OMML.XSL',
    ]
    xslt_path = None
    for p in candidates:
        if os.path.exists(p):
            xslt_path = p
            break
            
    if xslt_path is None:
        return None
        
    try:
        xslt_doc = ET.parse(xslt_path)
        _OMML_TRANSFORM = ET.XSLT(xslt_doc)
        return _OMML_TRANSFORM
    except Exception as e:
        print(f"Warning: Failed to load MML2OMML XSLT: {e}")
        return None

def latex_to_omml_element(latex_code):
    """
    Converts LaTeX math string to Word OMML XML element (<m:oMath>).
    Returns an oxml element if successful, or None on error.
    """
    try:
        from latex2mathml.converter import convert
        import lxml.etree as ET
        
        transform = _get_omml_transform()
        if transform is None:
            return None
            
        clean_latex = latex_code.strip()
        mathml_str = convert(clean_latex)
        tree = ET.fromstring(mathml_str.encode('utf-8'))
        omml_dom = transform(tree)
        xml_bytes = ET.tostring(omml_dom.getroot())
        return parse_xml(xml_bytes.decode('utf-8'))
    except Exception as ex:
        print(f"Warning: OMML conversion failed for '{latex_code[:30]}...': {ex}")
        return None

def add_equation(doc, eq_text, eq_num_str):
    """
    Renders an equation in a 2-column callout table with subtle light background.
    Compiles LaTeX formula into native Word OMML (<m:oMath>) equation element.
    """
    tbl = doc.add_table(rows=1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    tblPr = tbl._tbl.tblPr
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
    tblPr.append(borders)
    
    row = tbl.rows[0]
    cell_eq = row.cells[0]
    cell_num = row.cells[1]
    
    cell_eq.width = Cm(12.5)
    cell_num.width = Cm(2.0)
    cell_eq.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    cell_num.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    
    # Cell shading
    shd1 = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F8F9FA"/>')
    shd2 = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F8F9FA"/>')
    cell_eq._tc.get_or_add_tcPr().append(shd1)
    cell_num._tc.get_or_add_tcPr().append(shd2)
    
    p_eq = cell_eq.paragraphs[0]
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.line_spacing = 1.15
    p_eq.paragraph_format.space_before = Pt(4)
    p_eq.paragraph_format.space_after = Pt(4)
    
    omml_elem = latex_to_omml_element(eq_text)
    if omml_elem is not None:
        p_eq._p.append(omml_elem)
    else:
        # Fallback to plain text run if converter is unavailable
        r_eq = p_eq.add_run(eq_text)
        r_eq.font.name = 'Cambria Math'
        r_eq.font.size = Pt(11)
        r_eq.italic = True
    
    p_num = cell_num.paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.line_spacing = 1.15
    p_num.paragraph_format.space_before = Pt(4)
    p_num.paragraph_format.space_after = Pt(4)
    r_num = p_num.add_run(eq_num_str)
    r_num.font.name = 'Times New Roman'
    r_num.font.size = Pt(11)
    r_num.bold = True
    
    # Spacing after equation table
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(2)
    p_sp.paragraph_format.space_after = Pt(4)
    p_sp.paragraph_format.line_spacing = 1.0

def add_figure(doc, img_path, caption_text, fig_num_str, width_cm=14.2):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    p_img.paragraph_format.line_spacing = 1.0
    
    if os.path.exists(img_path):
        p_img.add_run().add_picture(img_path, width=Cm(width_cm))
    else:
        r_ph = p_img.add_run(f"[FIGURE PLACEHOLDER: {img_path}]")
        r_ph.font.name = 'Times New Roman'
        r_ph.font.size = Pt(11)
        r_ph.bold = True
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    p_cap.paragraph_format.line_spacing = 1.2
    
    r_lbl = p_cap.add_run(f"Figure {fig_num_str}: ")
    r_lbl.font.name = 'Times New Roman'
    r_lbl.font.size = Pt(10.5)
    r_lbl.bold = True
    
    _append_text_with_math(p_cap, caption_text, Pt(10.5))

def add_table_custom(doc, headers, data, caption_text, table_num_str, col_widths=None):
    # Caption above table
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_cap.paragraph_format.space_before = Pt(8)
    p_cap.paragraph_format.space_after = Pt(4)
    p_cap.paragraph_format.line_spacing = 1.2
    p_cap.paragraph_format.keep_with_next = True
    
    r_lbl = p_cap.add_run(f"Table {table_num_str}: ")
    r_lbl.font.name = 'Times New Roman'
    r_lbl.font.size = Pt(10.5)
    r_lbl.bold = True
    
    _append_text_with_math(p_cap, caption_text, Pt(10.5))
    
    num_rows = len(data) + 1
    num_cols = len(headers)
    tbl = doc.add_table(rows=num_rows, cols=num_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    # Styling table borders
    tblPr = tbl._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="2E5B82"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="2E5B82"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)
    
    # Headers
    hdr_row = tbl.rows[0]
    for c_idx, h_text in enumerate(headers):
        cell = hdr_row.cells[c_idx]
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="EBF2FA"/>')
        cell._tc.get_or_add_tcPr().append(shd)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.05
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(h_text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.bold = True
        
    # Data rows
    for r_idx, row_values in enumerate(data):
        row = tbl.rows[r_idx + 1]
        is_even = (r_idx % 2 == 1)
        fill_color = "F9FBFC" if is_even else "FFFFFF"
        for c_idx, val in enumerate(row_values):
            cell = row.cells[c_idx]
            if is_even:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_color}"/>')
                cell._tc.get_or_add_tcPr().append(shd)
            p = cell.paragraphs[0]
            # align first column left, others center or left depending on type
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.space_before = Pt(2.5)
            p.paragraph_format.space_after = Pt(2.5)
            _append_text_with_math(p, str(val), Pt(9.5))
            if "Ours" in str(val) or "RRFN" in str(val) or (c_idx == 0 and "RRFN" in str(row_values[0])):
                for r in p.runs:
                    r.bold = True
                
    if col_widths and len(col_widths) == num_cols:
        for r in tbl.rows:
            for c_idx, w in enumerate(col_widths):
                r.cells[c_idx].width = Cm(w)
                
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(2)
    p_sp.paragraph_format.space_after = Pt(6)
    p_sp.paragraph_format.line_spacing = 1.0

def add_callout_box(doc, text, title="IMPORTANT NOTICE"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.rows[0].cells[0]
    cell.width = Cm(14.5)
    
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F4F6F9"/>')
    cell._tc.get_or_add_tcPr().append(shd)
    
    tblPr = tbl._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="B0C4DE"/>'
        f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="B0C4DE"/>'
        f'  <w:left w:val="single" w:sz="18" w:space="0" w:color="1F4E79"/>'
        f'  <w:right w:val="single" w:sz="4" w:space="0" w:color="B0C4DE"/>'
        f'  <w:insideH w:val="none"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)
    
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    
    if title:
        r_t = p.add_run(f"[{title}]\n")
        r_t.font.name = 'Times New Roman'
        r_t.font.size = Pt(11)
        r_t.bold = True
        
    _append_text_with_math(p, text, Pt(10.5))
    
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(2)
    p_sp.paragraph_format.space_after = Pt(6)

def add_code_block(doc, code_str, caption=None):
    if caption:
        p_c = doc.add_paragraph()
        p_c.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_c.paragraph_format.space_before = Pt(4)
        p_c.paragraph_format.space_after = Pt(2)
        r_c = p_c.add_run(f"Listing: {caption}")
        r_c.font.name = 'Times New Roman'
        r_c.font.size = Pt(10)
        r_c.bold = True
        
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.rows[0].cells[0]
    cell.width = Cm(14.5)
    
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F8F9FA"/>')
    cell._tc.get_or_add_tcPr().append(shd)
    
    tblPr = tbl._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="E1E4E8"/>'
        f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="E1E4E8"/>'
        f'  <w:left w:val="single" w:sz="4" w:space="0" w:color="E1E4E8"/>'
        f'  <w:right w:val="single" w:sz="4" w:space="0" w:color="E1E4E8"/>'
        f'  <w:insideH w:val="none"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)
    
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    
    run = p.add_run(code_str)
    run.font.name = 'Consolas'
    run.font.size = Pt(9.0)
    
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(2)
    p_sp.paragraph_format.space_after = Pt(6)
