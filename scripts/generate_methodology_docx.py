from __future__ import annotations

import datetime as _dt
import html
import os
import zipfile
from pathlib import Path


def _xml_escape(text: str) -> str:
    return html.escape(text, quote=True)


def _p(text: str) -> str:
    """
    Un párrafo simple (WordprocessingML).
    """
    t = _xml_escape(text)
    return (
        "<w:p>"
        "<w:r>"
        f"<w:t xml:space=\"preserve\">{t}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def _h(text: str, level: int) -> str:
    """
    Encabezado. level 1..3.
    """
    lvl = max(1, min(3, int(level)))
    t = _xml_escape(text)
    return (
        "<w:p>"
        f"<w:pPr><w:pStyle w:val=\"Heading{lvl}\"/></w:pPr>"
        "<w:r>"
        f"<w:t xml:space=\"preserve\">{t}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def _bullets(items: list[str]) -> str:
    """
    Bullets simples sin depender de numbering.xml (usa el carácter '•').
    """
    parts = []
    for it in items:
        parts.append(_p(f"• {it}"))
    return "".join(parts)


def _doc_xml(paragraphs: list[str]) -> str:
    body = "".join(paragraphs) + "<w:sectPr/>"
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        f"<w:body>{body}</w:body>"
        "</w:document>"
    )


def _styles_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
        "<w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\">"
        "<w:name w:val=\"Normal\"/>"
        "</w:style>"
        "<w:style w:type=\"paragraph\" w:styleId=\"Heading1\">"
        "<w:name w:val=\"heading 1\"/>"
        "<w:basedOn w:val=\"Normal\"/>"
        "<w:qFormat/>"
        "<w:pPr><w:spacing w:before=\"240\" w:after=\"120\"/></w:pPr>"
        "<w:rPr><w:b/><w:sz w:val=\"32\"/></w:rPr>"
        "</w:style>"
        "<w:style w:type=\"paragraph\" w:styleId=\"Heading2\">"
        "<w:name w:val=\"heading 2\"/>"
        "<w:basedOn w:val=\"Normal\"/>"
        "<w:qFormat/>"
        "<w:pPr><w:spacing w:before=\"200\" w:after=\"100\"/></w:pPr>"
        "<w:rPr><w:b/><w:sz w:val=\"28\"/></w:rPr>"
        "</w:style>"
        "<w:style w:type=\"paragraph\" w:styleId=\"Heading3\">"
        "<w:name w:val=\"heading 3\"/>"
        "<w:basedOn w:val=\"Normal\"/>"
        "<w:qFormat/>"
        "<w:pPr><w:spacing w:before=\"160\" w:after=\"80\"/></w:pPr>"
        "<w:rPr><w:b/><w:sz w:val=\"24\"/></w:rPr>"
        "</w:style>"
        "</w:styles>"
    )


def _content_types_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "<Override PartName=\"/word/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/>"
        "<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>"
        "<Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>"
        "</Types>"
    )


def _rels_root_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"word/document.xml\"/>"
        "<Relationship Id=\"rId2\" "
        "Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" "
        "Target=\"docProps/core.xml\"/>"
        "<Relationship Id=\"rId3\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" "
        "Target=\"docProps/app.xml\"/>"
        "</Relationships>"
    )


def _rels_document_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" "
        "Target=\"styles.xml\"/>"
        "</Relationships>"
    )


def _core_xml(title: str, creator: str) -> str:
    created = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    t = _xml_escape(title)
    c = _xml_escape(creator)
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<cp:coreProperties "
        "xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
        "xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
        "xmlns:dcterms=\"http://purl.org/dc/terms/\" "
        "xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" "
        "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
        f"<dc:title>{t}</dc:title>"
        f"<dc:creator>{c}</dc:creator>"
        "<cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        f"<dcterms:created xsi:type=\"dcterms:W3CDTF\">{created}</dcterms:created>"
        f"<dcterms:modified xsi:type=\"dcterms:W3CDTF\">{created}</dcterms:modified>"
        "</cp:coreProperties>"
    )


def _app_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" "
        "xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">"
        "<Application>Microsoft Office Word</Application>"
        "</Properties>"
    )


def build_document(project_root: Path) -> list[str]:
    paragraphs: list[str] = []

    paragraphs.append(_h("Metodología: Tutor Inteligente de Matemáticas (KDD Cup 2010)", 1))
    paragraphs.append(_p("Documento técnico para describir el flujo completo desde datos crudos hasta prueba del modelo."))
    paragraphs.append(_p(f"Repositorio: {project_root}"))
    paragraphs.append(_p(f"Fecha: {_dt.date.today().isoformat()}"))

    paragraphs.append(_h("1. Objetivo", 2))
    paragraphs.append(
        _p(
            "Construir un tutor local que estime la dificultad observada del intento del estudiante "
            "(baja/media/alta) y aplique una política adaptativa (subir/mantener/bajar) basada en desempeño."
        )
    )

    paragraphs.append(_h("2. Datos (fuentes)", 2))
    paragraphs.append(_bullets([
        "KDD Cup 2010 (Algebra I): archivos .txt en data/external/ (train/test/master).",
        "Logs del juego: data/raw/gameplay_logs.csv (intentos reales; se agregan para reentrenar).",
    ]))

    paragraphs.append(_h("3. Normalización del dataset", 2))
    paragraphs.append(_p("Módulo: src/data_processing/data_loader.py"))
    paragraphs.append(_p("Se transforma el esquema KDD al esquema estándar del proyecto:"))
    paragraphs.append(_bullets([
        "student_id ← Anon Student Id (o Student ID si aplica).",
        "step_name ← Step Name.",
        "incorrects ← Incorrects.",
        "hints ← Hints.",
        "correct_first_attempt ← Correct First Attempt (0/1).",
        "step_duration_sec ← Step Duration (sec).",
        "timestamp ← First Transaction Time / Transaction Time (si existe).",
        "source_split ← train/test/master/user_gameplay.",
    ]))
    paragraphs.append(_p("Limpieza: coerción numérica segura, relleno de nulos, recorte de rangos inválidos (>=0), CFA en {0,1}."))

    paragraphs.append(_h("4. Feature engineering", 2))
    paragraphs.append(_p("Módulo: src/data_processing/feature_engineering.py"))
    paragraphs.append(_p("Se calculan features explicables en cuatro grupos:"))
    paragraphs.append(_h("4.1 Derivadas simples", 3))
    paragraphs.append(_bullets([
        "error_rate = incorrects / (incorrects + hints + 1).",
        "time_efficiency = correct_first_attempt / (step_duration_sec + 1).",
    ]))
    paragraphs.append(_h("4.2 Orden temporal (event_order)", 3))
    paragraphs.append(_bullets([
        "Si timestamp es parseable: usa su orden temporal.",
        "Si no: usa el índice/orden original como fallback.",
        "Este orden se usa para evitar fuga de información al crear promedios 'prev'.",
    ]))
    paragraphs.append(_h("4.3 Históricas 'prev' (solo pasado)", 3))
    paragraphs.append(_bullets([
        "student_attempt_count_prev: número de intentos previos del estudiante.",
        "student_avg_incorrects_prev: promedio previo de incorrects por estudiante.",
        "student_avg_time_prev: promedio previo de tiempo por estudiante.",
        "student_accuracy_prev: promedio previo de correct_first_attempt por estudiante.",
        "step_success_rate_prev: tasa previa de acierto para ese step_name.",
    ]))
    paragraphs.append(_h("4.4 Estructurales desde step_name", 3))
    paragraphs.append(_bullets([
        "step_len: longitud del texto.",
        "step_num_ops: conteo de operadores (+ - * / =).",
        "step_has_parentheses: 1 si contiene paréntesis.",
        "step_num_digits: conteo de dígitos.",
        "step_num_variables: conteo de letras.",
        "step_abs_constant_sum: suma abs. de constantes detectadas (con filtro anti-outliers).",
    ]))

    paragraphs.append(_h("5. Construcción del target (difficulty_level)", 2))
    paragraphs.append(_p("Se calcula difficulty_score (solo para construir el target) y se discretiza por cuantiles en 3 clases:"))
    paragraphs.append(_bullets([
        "0 = baja, 1 = media, 2 = alta.",
        "difficulty_score NO se usa como feature de entrada al modelo (evita leakage directo).",
    ]))

    paragraphs.append(_h("6. Entrenamiento y validación", 2))
    paragraphs.append(_p("Módulo: src/models/model_trainer.py"))
    paragraphs.append(_p("Modelos entrenados en el pipeline del tutor: random_forest, gradient_boosting, logistic_regression."))
    paragraphs.append(_p("La métrica de selección interna es F1-weighted."))
    paragraphs.append(_p("Esquemas de validación (prioridad):"))
    paragraphs.append(_bullets([
        "fixed_train_test_split si source_split=train/test existe y el test tiene variación.",
        "group_student_holdout_split: holdout por student_id (evita que el mismo estudiante aparezca en train y test).",
        "temporal_holdout_split si existe event_order y hay variación.",
        "random_stratified_split / random_split como último recurso.",
    ]))

    paragraphs.append(_h("7. Artefactos y reproducibilidad", 2))
    paragraphs.append(_bullets([
        "Modelo final: models/model.pkl",
        "Leaderboard: reports/metrics/model_leaderboard.csv",
        "Resumen: reports/metrics/training_summary.json",
        "Dataset procesado (para auditoría): reports/metrics/processed_dataset.csv",
    ]))

    paragraphs.append(_h("8. Inferencia (API) y trazabilidad", 2))
    paragraphs.append(_p("Módulo: src/api/main.py + src/models/difficulty_model.py"))
    paragraphs.append(_bullets([
        "La API construye el record del intento (errores/pistas/tiempo/CFA) y llama al modelo.",
        "probability = max(predict_proba) del clasificador (confianza de la clase predicha).",
        "Se guarda model_inputs como snapshot de features (auditoría).",
        "Se registra cada intento en data/raw/gameplay_logs.csv (incluye model_input_* y la recomendación).",
    ]))

    paragraphs.append(_h("9. Política adaptativa post-intento", 2))
    paragraphs.append(_p("Módulo: src/app/adaptive_engine.py"))
    paragraphs.append(_bullets([
        "Calcula performance_score (0..1) con tiempo/errores/pistas.",
        "Decisión robusta con ventana corta (anti-oscilación) sobre los últimos intentos.",
        "Acciones: subir/mantener/bajar; además etiqueta de política (policy_label).",
    ]))

    paragraphs.append(_h("10. Métricas de aprendizaje (panel admin)", 2))
    paragraphs.append(_p("Módulo: src/app/learning_metrics.py (a partir de data/raw/gameplay_logs.csv)"))
    paragraphs.append(_bullets([
        "Accuracy global y por usuario.",
        "Tiempo/pistas/errores promedio.",
        "Dependencia de pistas en intentos correctos.",
        "Intentos promedio hasta resolver.",
        "Tasa de oscilación del nivel.",
        "Deltas (2ª mitad - 1ª mitad) para tendencia por usuario.",
    ]))

    paragraphs.append(_h("11. Procedimiento de prueba del modelo", 2))
    paragraphs.append(_bullets([
        "1) Entrenar modelo: python main.py train --no-plots",
        "2) Verificar artefactos en reports/metrics/ y models/model.pkl",
        "3) Ejecutar prueba rápida: python test_model.py",
        "4) Levantar API: python main.py serve y probar /docs (Swagger) y el panel admin.",
        "5) Jugar varias rondas para generar gameplay_logs.csv y revisar métricas de aprendizaje.",
    ]))

    paragraphs.append(_h("12. Notas y buenas prácticas", 2))
    paragraphs.append(_bullets([
        "Tratar outliers en tiempo/errores/pistas al analizar (mediana y percentiles son más robustos que la media).",
        "Evitar leakage: no incluir difficulty_score como feature (ya corregido).",
        "Evaluar el tutor por impacto en logs (tendencias) además de métricas del clasificador.",
    ]))

    return paragraphs


def write_docx(output_path: Path, paragraphs: list[str], title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    creator = os.environ.get("USERNAME") or os.environ.get("USER") or "Autor"

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml())
        zf.writestr("_rels/.rels", _rels_root_xml())
        zf.writestr("word/document.xml", _doc_xml(paragraphs))
        zf.writestr("word/_rels/document.xml.rels", _rels_document_xml())
        zf.writestr("word/styles.xml", _styles_xml())
        zf.writestr("docProps/core.xml", _core_xml(title=title, creator=creator))
        zf.writestr("docProps/app.xml", _app_xml())


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paragraphs = build_document(root)
    out = root / "reports" / "Metodologia_Tutor_Inteligente.docx"
    write_docx(out, paragraphs, title="Metodología - Tutor Inteligente")
    print(f"Generado: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

