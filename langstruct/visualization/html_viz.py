"""HTML visualization utilities for extraction results."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..core.export_utils import ExportUtilities
from ..core.schemas import ExtractionResult, SourceSpan


def visualize(
    results: Union[str, Path, List[ExtractionResult]],
    title: str = "LangStruct Extraction Results",
    show_confidence: bool = True,
    show_sources: bool = True,
    highlight_color: str = "#ffffcc",
) -> str:
    """Generate interactive HTML visualization of extraction results.

    This creates an interactive HTML page showing extracted entities with
    source highlighting, similar to LangExtract's visualization capabilities.

    Args:
        results: Either path to JSONL file or list of ExtractionResult objects
        title: Title for the HTML page
        show_confidence: Whether to display confidence scores
        show_sources: Whether to highlight source locations
        highlight_color: CSS color for source highlighting

    Returns:
        HTML string ready for saving or displaying

    Example:
        >>> results = extractor.extract(texts)
        >>> html = visualize(results, title="My Extraction Results")
        >>> with open("results.html", "w") as f:
        ...     f.write(html)
    """
    # Load results if path provided
    if isinstance(results, (str, Path)):
        results = ExportUtilities.load_annotated_documents(results)

    if not results:
        return _generate_empty_html(title)

    # Generate HTML content
    html_content = _generate_html_template(
        results, title, show_confidence, show_sources, highlight_color
    )

    return html_content


def save_visualization(
    results: Union[str, Path, List[ExtractionResult]],
    file_path: Union[str, Path],
    **kwargs,
) -> None:
    """Save visualization to HTML file.

    Args:
        results: Either path to JSONL file or list of ExtractionResult objects
        file_path: Path to save HTML file
        **kwargs: Additional arguments for visualize()
    """
    html_content = visualize(results, **kwargs)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)


class HTMLVisualizer:
    """Generate interactive HTML visualizations of extraction results."""

    def __init__(self):
        self.template = self._get_html_template()

    def visualize(
        self,
        text: str,
        result: ExtractionResult,
        title: str = "LangStruct Extraction Results",
    ) -> str:
        """Create HTML visualization of extraction with source highlighting.

        Args:
            text: Original text that was processed
            result: Extraction result with entities and sources
            title: Title for the HTML page

        Returns:
            HTML string with interactive visualization
        """
        # Create highlighted text with source spans
        highlighted_text = self._create_highlighted_text(
            text, result.sources, result.entities
        )

        # Format entities for display
        entities_html = self._format_entities(result.entities, result.sources)

        # Calculate stats
        entity_count = len([v for v in result.entities.values() if v is not None])
        source_count = sum(len(spans) for spans in result.sources.values())

        # Generate HTML
        html = self.template.format(
            title=title,
            highlighted_text=highlighted_text,
            entities_html=entities_html,
            confidence=f"{result.confidence:.1%}",
            metadata_json=json.dumps(result.metadata, indent=2),
            entity_count=entity_count,
            source_count=source_count,
        )

        return html

    def save_visualization(
        self,
        text: str,
        result: ExtractionResult,
        output_path: str,
        title: str = "LangStruct Extraction Results",
    ) -> None:
        """Save HTML visualization to file.

        Args:
            text: Original text that was processed
            result: Extraction result with entities and sources
            output_path: Path to save HTML file
            title: Title for the HTML page
        """
        html = self.visualize(text, result, title)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _create_highlighted_text(
        self,
        text: str,
        sources: Dict[str, List[SourceSpan]],
        entities: Optional[Dict[str, any]] = None,
    ) -> str:
        """Create HTML with highlighted source spans."""
        if not sources:
            return f"<p>{self._escape_html(text)}</p>"

        # Collect all spans with their field names
        all_spans = []
        colors = self._get_field_colors()
        color_idx = 0

        for field_name, spans in sources.items():
            color = colors[color_idx % len(colors)]
            color_idx += 1

            for span in spans:
                # Validate span boundaries and use fallback if invalid
                start = max(0, min(span.start, len(text)))
                end = max(start, min(span.end, len(text)))

                # Check if the span actually contains meaningful text
                span_is_valid = False
                if start < end and end <= len(text):
                    actual_span_text = text[start:end].strip()
                    # Check if span text matches entity value or is reasonable
                    if entities and field_name in entities:
                        entity_str = str(entities[field_name]).strip()
                        if (
                            entity_str.lower() in actual_span_text.lower()
                            or actual_span_text.lower() in entity_str.lower()
                        ):
                            span_is_valid = True
                    # Or if span text is provided and matches
                    elif (
                        span.text
                        and span.text.strip()
                        and span.text.strip().lower() in actual_span_text.lower()
                    ):
                        span_is_valid = True

                # If span is invalid, try to find the text in the document
                if not span_is_valid:
                    search_text = None

                    # Try span text first
                    if span.text and span.text.strip():
                        search_text = span.text.strip()
                    # Fallback to entity value if span text is empty
                    elif entities and field_name in entities:
                        entity_value = str(entities[field_name]).strip()
                        if entity_value:
                            search_text = entity_value

                    # Search for the text in the document
                    if search_text:
                        found_idx = text.find(search_text)
                        if found_idx >= 0:
                            start = found_idx
                            end = found_idx + len(search_text)
                        else:
                            continue  # Skip spans we can't locate
                    else:
                        continue

                # Only include valid spans
                if start < end and end <= len(text):
                    all_spans.append(
                        {
                            "start": start,
                            "end": end,
                            "field": field_name,
                            "color": color,
                            "text": span.text or text[start:end],
                        }
                    )

        # Sort spans by start position
        all_spans.sort(key=lambda x: x["start"])

        # Build highlighted HTML
        result_html = []
        last_pos = 0

        for span in all_spans:
            # Add text before this span
            if span["start"] > last_pos:
                result_html.append(self._escape_html(text[last_pos : span["start"]]))

            # Add highlighted span
            result_html.append(
                f'<span class="highlight" style="background-color: {span["color"]}" '
                f'data-field="{span["field"]}" data-text="{self._escape_html(span["text"])}" '
                f'title="{span["field"]}: {self._escape_html(span["text"])}">'
                f'{self._escape_html(text[span["start"]:span["end"]])}'
                f"</span>"
            )

            last_pos = span["end"]

        # Add remaining text
        if last_pos < len(text):
            result_html.append(self._escape_html(text[last_pos:]))

        return f'<p>{"".join(result_html)}</p>'

    def _format_entities(
        self, entities: Dict[str, any], sources: Dict[str, List[SourceSpan]]
    ) -> str:
        """Format entities for HTML display."""
        if not entities:
            return "<p><em>No entities extracted</em></p>"

        html_parts = []
        colors = self._get_field_colors()
        color_idx = 0

        for field_name, value in entities.items():
            color = colors[color_idx % len(colors)]
            color_idx += 1

            # Format the value
            if isinstance(value, (list, dict)):
                value_str = json.dumps(value, indent=2)
            else:
                value_str = str(value) if value is not None else "None"

            # Get source count
            source_count = len(sources.get(field_name, []))

            html_parts.append(
                f"""
                <div class="entity-item">
                    <div class="entity-field">
                        {field_name}
                        {f'<span class="source-count">{source_count} source{"s" if source_count != 1 else ""}</span>' if source_count > 0 else ''}
                    </div>
                    <div class="entity-value">{self._escape_html(value_str)}</div>
                </div>
            """
            )

        return "".join(html_parts)

    def _get_field_colors(self) -> List[str]:
        """Get list of modern colors for highlighting different fields."""
        return [
            "rgba(239, 68, 68, 0.15)",  # Modern red
            "rgba(59, 130, 246, 0.15)",  # Modern blue
            "rgba(34, 197, 94, 0.15)",  # Modern green
            "rgba(249, 115, 22, 0.15)",  # Modern orange
            "rgba(168, 85, 247, 0.15)",  # Modern purple
            "rgba(20, 184, 166, 0.15)",  # Modern teal
            "rgba(236, 72, 153, 0.15)",  # Modern pink
            "rgba(132, 204, 22, 0.15)",  # Modern lime
        ]

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    def _get_html_template(self) -> str:
        """Get the HTML template for visualization."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            padding: 40px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        h1 {{
            color: #2d3748;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 40px;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
            padding: 0;
        }}
        
        .stat {{
            text-align: center;
            padding: 25px 20px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }}
        
        .stat:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        }}
        
        .stat-value {{
            font-size: 2.2em;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }}
        
        .stat-label {{
            color: #64748b;
            font-size: 0.9em;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .entities {{
            margin-bottom: 40px;
            display: grid;
            gap: 15px;
        }}
        
        .entity-item {{
            background: rgba(255, 255, 255, 0.9);
            border-left: 4px solid #667eea;
            margin-bottom: 0;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }}
        
        .entity-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border-left-color: #764ba2;
        }}
        
        .entity-field {{
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 8px;
            text-transform: capitalize;
            font-size: 1.1em;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        
        .entity-value {{
            color: #4a5568;
            font-size: 1.05em;
            line-height: 1.4;
        }}
        
        .source-count {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 600;
            margin-left: 10px;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }}
        
        .text-content {{
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 30px;
            font-size: 16px;
            line-height: 1.8;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            backdrop-filter: blur(10px);
            margin-bottom: 30px;
        }}
        
        .highlight {{
            padding: 3px 6px;
            border-radius: 8px;
            border: 2px solid rgba(255, 255, 255, 0.5);
            font-weight: 500;
            transition: all 0.2s ease;
            cursor: pointer;
            position: relative;
        }}
        
        .highlight:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
            z-index: 10;
            cursor: help;
        }}
        
        .highlight:hover {{
            background-color: #ffc107;
            color: white;
        }}
        
        .metadata {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 25px;
            margin-top: 30px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .metadata h3 {{
            color: #2d3748;
            font-size: 1.2em;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .metadata pre {{
            background: rgba(248, 250, 252, 0.8);
            border-radius: 8px;
            padding: 15px;
            font-size: 0.85em;
            color: #475569;
            overflow-x: auto;
            border: 1px solid rgba(226, 232, 240, 0.5);
            margin: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{entity_count}</div>
                <div class="stat-label">Entities Extracted</div>
            </div>
            <div class="stat">
                <div class="stat-value">{source_count}</div>
                <div class="stat-label">Source Spans</div>
            </div>
            <div class="stat">
                <div class="stat-value">{confidence}</div>
                <div class="stat-label">Confidence</div>
            </div>
        </div>
        
        <div class="entities">
            {entities_html}
        </div>
        
        <div class="text-content">
            {highlighted_text}
        </div>
        
        <div class="metadata">
            <h3>Metadata</h3>
            <pre>{metadata_json}</pre>
        </div>
    </div>
</body>
</html>
        """


def _generate_empty_html(title: str) -> str:
    """Generate HTML for empty results."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .empty {{ text-align: center; color: #666; padding: 40px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="empty">
            <p>No extraction results to display.</p>
        </div>
    </div>
</body>
</html>
    """


def _generate_html_template(
    results: List[ExtractionResult],
    title: str,
    show_confidence: bool,
    show_sources: bool,
    highlight_color: str,
) -> str:
    """Generate the enhanced interactive HTML template with LangExtract-style features."""

    # Prepare data for JavaScript with text content
    results_data = []
    all_field_types = set()

    for i, result in enumerate(results):
        # Get original text from metadata if available
        original_text = result.metadata.get("original_text", "")

        result_data = {
            "id": i,
            "entities": result.entities,
            "confidence": result.confidence,
            "sources": {},
            "metadata": result.metadata,
            "text": original_text,
        }

        # Convert SourceSpan objects to dictionaries and collect field types
        if show_sources and result.sources:
            for field_name, spans in result.sources.items():
                all_field_types.add(field_name)
                result_data["sources"][field_name] = [
                    {"start": span.start, "end": span.end, "text": span.text}
                    for span in spans
                ]
        else:
            # Add field types even without sources
            for field_name in result.entities.keys():
                all_field_types.add(field_name)

        results_data.append(result_data)

    # Generate color mapping for consistent field coloring
    field_colors = _generate_field_color_mapping(list(all_field_types))

    # Generate the enhanced HTML
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding: 20px 0;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .header-content {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }}
        
        h1 {{
            color: #333;
            margin: 0;
            font-size: 2em;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .controls {{
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .search-box {{
            position: relative;
        }}
        
        .search-input {{
            padding: 8px 15px 8px 40px;
            border: 2px solid #e1e5e9;
            border-radius: 25px;
            font-size: 14px;
            width: 250px;
            transition: all 0.3s ease;
        }}
        
        .search-input:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        
        .search-icon {{
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: #999;
        }}
        
        .filter-chips {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        
        .filter-chip {{
            padding: 6px 12px;
            border: 2px solid #e1e5e9;
            border-radius: 20px;
            background: white;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
            user-select: none;
        }}
        
        .filter-chip.active {{
            border-color: #667eea;
            background: #667eea;
            color: white;
        }}
        
        .filter-chip:hover:not(.active) {{
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }}
        
        .view-toggle {{
            display: flex;
            background: white;
            border-radius: 25px;
            border: 2px solid #e1e5e9;
            overflow: hidden;
        }}
        
        .view-btn {{
            padding: 8px 15px;
            border: none;
            background: transparent;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .view-btn.active {{
            background: #667eea;
            color: white;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 30px;
        }}
        
        .main-content {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .sidebar {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .stats-panel {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            gap: 15px;
        }}
        
        .stat {{
            text-align: center;
            padding: 15px 10px;
            border-radius: 10px;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 0.85em;
            color: #666;
            font-weight: 500;
        }}
        
        .entity-legend {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .legend-title {{
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
            padding: 5px;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s ease;
        }}
        
        .legend-item:hover {{
            background: rgba(102, 126, 234, 0.05);
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 2px solid rgba(0, 0, 0, 0.1);
        }}
        
        .legend-label {{
            font-size: 0.9em;
            font-weight: 500;
            color: #555;
        }}
        
        .legend-count {{
            margin-left: auto;
            font-size: 0.8em;
            color: #999;
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 10px;
        }}
        
        .document-text {{
            background: white;
            border: 1px solid #e1e5e9;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            font-size: 16px;
            line-height: 1.8;
            font-family: 'Georgia', serif;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.02);
        }}
        
        .highlighted-entity {{
            padding: 2px 4px;
            border-radius: 4px;
            cursor: pointer;
            position: relative;
            border: 2px solid transparent;
            transition: all 0.2s ease;
            display: inline-block;
        }}
        
        .highlighted-entity:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            border-color: rgba(0, 0, 0, 0.2);
        }}
        
        .highlighted-entity.selected {{
            border-color: #333;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
        }}
        
        .entity-tooltip {{
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            white-space: nowrap;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: all 0.2s ease;
            margin-bottom: 5px;
        }}
        
        .entity-tooltip::after {{
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 5px solid transparent;
            border-top-color: #333;
        }}
        
        .highlighted-entity:hover .entity-tooltip {{
            opacity: 1;
            visibility: visible;
        }}
        
        .entities-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        
        .entity-summary-card {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 15px;
            transition: all 0.2s ease;
        }}
        
        .entity-summary-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .entity-summary-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        
        .entity-summary-field {{
            font-weight: 600;
            color: #495057;
            font-size: 0.9em;
            text-transform: capitalize;
        }}
        
        .entity-summary-value {{
            color: #6c757d;
            font-size: 0.95em;
            line-height: 1.4;
        }}
        
        .entity-count-badge {{
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 500;
        }}
        
        .no-results {{
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }}
        
        .no-results-icon {{
            font-size: 3em;
            margin-bottom: 15px;
        }}
        
        @media (max-width: 1200px) {{
            .container {{
                grid-template-columns: 1fr;
                gap: 20px;
            }}
            
            .header-content {{
                padding: 0 20px;
            }}
        }}
        
        @media (max-width: 768px) {{
            .controls {{
                justify-content: center;
                width: 100%;
            }}
            
            .search-input {{
                width: 200px;
            }}
            
            .container {{
                padding: 20px;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1>{title}</h1>
            <div class="controls">
                <div class="search-box">
                    <span class="search-icon">🔍</span>
                    <input type="text" class="search-input" id="search-input" placeholder="Search entities or text...">
                </div>
                <div class="filter-chips" id="filter-chips">
                    <!-- Filter chips will be populated by JavaScript -->
                </div>
                <div class="view-toggle">
                    <button class="view-btn active" data-view="document">Document</button>
                    <button class="view-btn" data-view="list">List</button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="main-content">
            <div id="document-view">
                <!-- Document content will be populated by JavaScript -->
            </div>
            <div id="list-view" style="display: none;">
                <!-- List view content will be populated by JavaScript -->
            </div>
        </div>
        
        <div class="sidebar">
            <div class="stats-panel">
                <h3 style="margin: 0 0 15px 0; color: #333; font-size: 1.1em;">Statistics</h3>
                <div class="stats-grid">
                    <div class="stat">
                        <div class="stat-value" id="total-documents">0</div>
                        <div class="stat-label">Documents</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="total-entities">0</div>
                        <div class="stat-label">Entities</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="avg-confidence">0%</div>
                        <div class="stat-label">Avg Confidence</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="entity-types">0</div>
                        <div class="stat-label">Types</div>
                    </div>
                </div>
            </div>
            
            <div class="entity-legend">
                <h3 class="legend-title">Entity Types</h3>
                <div id="legend-items">
                    <!-- Legend items will be populated by JavaScript -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data from Python
        const results = {json.dumps(results_data, indent=2)};
        const showConfidence = {json.dumps(show_confidence)};
        const showSources = {json.dumps(show_sources)};
        const fieldColors = {json.dumps(field_colors, indent=2)};
        
        // Global state
        let currentView = 'document';
        let activeFilters = new Set();
        let searchQuery = '';
        let selectedEntity = null;
        
        // Initialize the visualization
        document.addEventListener('DOMContentLoaded', function() {{
            initializeFilters();
            setupEventListeners();
            renderVisualization();
            updateStats();
        }});
        
        function initializeFilters() {{
            const filterChips = document.getElementById('filter-chips');
            const fieldTypes = Object.keys(fieldColors);
            
            fieldTypes.forEach(fieldType => {{
                const chip = document.createElement('div');
                chip.className = 'filter-chip';
                chip.dataset.field = fieldType;
                chip.textContent = fieldType.replace('_', ' ').toUpperCase();
                chip.style.borderColor = fieldColors[fieldType];
                chip.addEventListener('click', () => toggleFilter(fieldType));
                filterChips.appendChild(chip);
            }});
        }}
        
        function setupEventListeners() {{
            // Search functionality
            document.getElementById('search-input').addEventListener('input', (e) => {{
                searchQuery = e.target.value.toLowerCase();
                renderVisualization();
            }});
            
            // View toggle
            document.querySelectorAll('.view-btn').forEach(btn => {{
                btn.addEventListener('click', (e) => {{
                    document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    currentView = e.target.dataset.view;
                    toggleView();
                }});
            }});
            
            // Legend interactions
            document.addEventListener('click', (e) => {{
                if (e.target.classList.contains('legend-item')) {{
                    const fieldType = e.target.dataset.field;
                    if (fieldType) toggleFilter(fieldType);
                }}
            }});
        }}
        
        function toggleFilter(fieldType) {{
            if (activeFilters.has(fieldType)) {{
                activeFilters.delete(fieldType);
            }} else {{
                activeFilters.add(fieldType);
            }}
            
            // Update chip appearance
            const chip = document.querySelector(`[data-field="${{fieldType}}"]`);
            if (chip) {{
                chip.classList.toggle('active', activeFilters.has(fieldType));
            }}
            
            renderVisualization();
        }}
        
        function toggleView() {{
            const documentView = document.getElementById('document-view');
            const listView = document.getElementById('list-view');
            
            if (currentView === 'document') {{
                documentView.style.display = 'block';
                listView.style.display = 'none';
            }} else {{
                documentView.style.display = 'none';
                listView.style.display = 'block';
            }}
        }}
        
        function renderVisualization() {{
            if (currentView === 'document') {{
                renderDocumentView();
            }} else {{
                renderListView();
            }}
            updateLegend();
        }}
        
        function renderDocumentView() {{
            const container = document.getElementById('document-view');
            container.innerHTML = '';
            
            results.forEach((result, index) => {{
                if (!matchesFilters(result) || !matchesSearch(result)) return;
                
                const docDiv = document.createElement('div');
                docDiv.className = 'fade-in';
                docDiv.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h3 style="margin: 0; color: #333;">Document ${{index + 1}}</h3>
                        ${{showConfidence ? `<span class="entity-count-badge">${{(result.confidence * 100).toFixed(1)}}% confidence</span>` : ''}}
                    </div>
                    <div class="entities-summary" id="entities-summary-${{index}}"></div>
                    <div class="document-text" id="document-text-${{index}}"></div>
                `;
                
                container.appendChild(docDiv);
                
                // Render entities summary
                renderEntitiesSummary(result, index);
                
                // Render highlighted text
                renderHighlightedText(result, index);
            }});
            
            if (container.children.length === 0) {{
                container.innerHTML = `
                    <div class="no-results">
                        <div class="no-results-icon">🔍</div>
                        <h3>No results found</h3>
                        <p>Try adjusting your search or filters</p>
                    </div>
                `;
            }}
        }}
        
        function renderEntitiesSummary(result, index) {{
            const container = document.getElementById(`entities-summary-${{index}}`);
            
            Object.entries(result.entities).forEach(([fieldName, value]) => {{
                if (!shouldShowField(fieldName)) return;
                
                const sourceCount = result.sources[fieldName] ? result.sources[fieldName].length : 0;
                const color = fieldColors[fieldName] || '#f8f9fa';
                
                const card = document.createElement('div');
                card.className = 'entity-summary-card';
                card.style.borderLeft = `4px solid ${{color}}`;
                
                card.innerHTML = `
                    <div class="entity-summary-header">
                        <div class="legend-color" style="background-color: ${{color}};"></div>
                        <span class="entity-summary-field">${{fieldName.replace('_', ' ')}}</span>
                        ${{sourceCount > 0 ? `<span class="entity-count-badge">${{sourceCount}}</span>` : ''}}
                    </div>
                    <div class="entity-summary-value">${{formatValue(value)}}</div>
                `;
                
                container.appendChild(card);
            }});
        }}
        
        function renderHighlightedText(result, index) {{
            const container = document.getElementById(`document-text-${{index}}`);
            const text = result.text || 'No text available';
            
            if (!result.sources || Object.keys(result.sources).length === 0) {{
                container.textContent = text;
                return;
            }}
            
            // Collect all spans
            const spans = [];
            Object.entries(result.sources).forEach(([fieldName, sourceSpans]) => {{
                if (!shouldShowField(fieldName)) return;
                
                sourceSpans.forEach(span => {{
                    spans.push({{
                        start: span.start,
                        end: span.end,
                        field: fieldName,
                        text: span.text,
                        color: fieldColors[fieldName] || '#ffffcc'
                    }});
                }});
            }});
            
            // Sort by start position
            spans.sort((a, b) => a.start - b.start);
            
            // Build highlighted HTML
            let html = '';
            let lastPos = 0;
            
            spans.forEach((span, spanIndex) => {{
                // Add text before span
                if (span.start > lastPos) {{
                    html += escapeHtml(text.substring(lastPos, span.start));
                }}
                
                // Add highlighted span
                html += `
                    <span class="highlighted-entity" 
                          style="background-color: ${{span.color}};"
                          data-field="${{span.field}}"
                          data-span-id="${{spanIndex}}"
                          onclick="selectEntity('${{span.field}}', ${{spanIndex}})">
                        ${{escapeHtml(span.text)}}
                        <div class="entity-tooltip">
                            ${{span.field.replace('_', ' ')}}: ${{escapeHtml(span.text)}}
                        </div>
                    </span>
                `;
                
                lastPos = span.end;
            }});
            
            // Add remaining text
            if (lastPos < text.length) {{
                html += escapeHtml(text.substring(lastPos));
            }}
            
            container.innerHTML = html;
        }}
        
        function renderListView() {{
            const container = document.getElementById('list-view');
            container.innerHTML = '';
            
            const allEntities = [];
            
            results.forEach((result, docIndex) => {{
                if (!matchesFilters(result) || !matchesSearch(result)) return;
                
                Object.entries(result.entities).forEach(([fieldName, value]) => {{
                    if (!shouldShowField(fieldName)) return;
                    
                    const sources = result.sources[fieldName] || [];
                    allEntities.push({{
                        docIndex,
                        field: fieldName,
                        value,
                        sources,
                        confidence: result.confidence
                    }});
                }});
            }});
            
            if (allEntities.length === 0) {{
                container.innerHTML = `
                    <div class="no-results">
                        <div class="no-results-icon">📋</div>
                        <h3>No entities found</h3>
                        <p>Try adjusting your search or filters</p>
                    </div>
                `;
                return;
            }}
            
            // Group by field type
            const groupedEntities = {{}};
            allEntities.forEach(entity => {{
                if (!groupedEntities[entity.field]) {{
                    groupedEntities[entity.field] = [];
                }}
                groupedEntities[entity.field].push(entity);
            }});
            
            Object.entries(groupedEntities).forEach(([fieldName, entities]) => {{
                const section = document.createElement('div');
                section.className = 'fade-in';
                section.style.marginBottom = '30px';
                
                const color = fieldColors[fieldName] || '#f8f9fa';
                
                section.innerHTML = `
                    <h3 style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px; color: #333;">
                        <div class="legend-color" style="background-color: ${{color}}; width: 20px; height: 20px;"></div>
                        ${{fieldName.replace('_', ' ').toUpperCase()}}
                        <span class="entity-count-badge">${{entities.length}}</span>
                    </h3>
                    <div class="entities-summary" id="field-entities-${{fieldName}}"></div>
                `;
                
                const entitiesContainer = section.querySelector(`#field-entities-${{fieldName}}`);
                
                entities.forEach(entity => {{
                    const card = document.createElement('div');
                    card.className = 'entity-summary-card';
                    card.style.borderLeft = `4px solid ${{color}}`;
                    
                    card.innerHTML = `
                        <div class="entity-summary-header">
                            <span class="entity-summary-field">Doc ${{entity.docIndex + 1}}</span>
                            ${{entity.sources.length > 0 ? `<span class="entity-count-badge">${{entity.sources.length}} source${{entity.sources.length !== 1 ? 's' : ''}}</span>` : ''}}
                            ${{showConfidence ? `<span class="legend-count">${{(entity.confidence * 100).toFixed(1)}}%</span>` : ''}}
                        </div>
                        <div class="entity-summary-value">${{formatValue(entity.value)}}</div>
                    `;
                    
                    entitiesContainer.appendChild(card);
                }});
                
                container.appendChild(section);
            }});
        }}
        
        function updateLegend() {{
            const container = document.getElementById('legend-items');
            container.innerHTML = '';
            
            // Count entities by type across all results
            const typeCounts = {{}};
            
            results.forEach(result => {{
                if (!matchesFilters(result) || !matchesSearch(result)) return;
                
                Object.keys(result.entities).forEach(fieldName => {{
                    if (!shouldShowField(fieldName)) return;
                    typeCounts[fieldName] = (typeCounts[fieldName] || 0) + 1;
                }});
            }});
            
            Object.entries(typeCounts).forEach(([fieldName, count]) => {{
                const item = document.createElement('div');
                item.className = 'legend-item';
                item.dataset.field = fieldName;
                item.style.backgroundColor = activeFilters.has(fieldName) ? 'rgba(102, 126, 234, 0.1)' : 'transparent';
                
                const color = fieldColors[fieldName] || '#f8f9fa';
                
                item.innerHTML = `
                    <div class="legend-color" style="background-color: ${{color}};"></div>
                    <span class="legend-label">${{fieldName.replace('_', ' ')}}</span>
                    <span class="legend-count">${{count}}</span>
                `;
                
                container.appendChild(item);
            }});
        }}
        
        function updateStats() {{
            const filteredResults = results.filter(result => matchesFilters(result) && matchesSearch(result));
            
            let totalEntities = 0;
            let totalConfidence = 0;
            const entityTypes = new Set();
            
            filteredResults.forEach(result => {{
                Object.keys(result.entities).forEach(fieldName => {{
                    if (shouldShowField(fieldName)) {{
                        totalEntities++;
                        entityTypes.add(fieldName);
                    }}
                }});
                totalConfidence += result.confidence;
            }});
            
            document.getElementById('total-documents').textContent = filteredResults.length;
            document.getElementById('total-entities').textContent = totalEntities;
            document.getElementById('avg-confidence').textContent = filteredResults.length > 0 ? 
                `${{((totalConfidence / filteredResults.length) * 100).toFixed(1)}}%` : '0%';
            document.getElementById('entity-types').textContent = entityTypes.size;
        }}
        
        function matchesFilters(result) {{
            if (activeFilters.size === 0) return true;
            
            return Object.keys(result.entities).some(fieldName => 
                activeFilters.has(fieldName)
            );
        }}
        
        function matchesSearch(result) {{
            if (!searchQuery) return true;
            
            // Search in entity values
            const entityMatch = Object.values(result.entities).some(value => 
                String(value).toLowerCase().includes(searchQuery)
            );
            
            // Search in text content
            const textMatch = result.text && result.text.toLowerCase().includes(searchQuery);
            
            return entityMatch || textMatch;
        }}
        
        function shouldShowField(fieldName) {{
            if (activeFilters.size === 0) return true;
            return activeFilters.has(fieldName);
        }}
        
        function selectEntity(fieldName, spanIndex) {{
            // Remove previous selection
            document.querySelectorAll('.highlighted-entity.selected').forEach(el => {{
                el.classList.remove('selected');
            }});
            
            // Add selection to clicked entity
            const entity = document.querySelector(`[data-field="${{fieldName}}"][data-span-id="${{spanIndex}}"]`);
            if (entity) {{
                entity.classList.add('selected');
                selectedEntity = {{ fieldName, spanIndex }};
            }}
        }}
        
        function formatValue(value) {{
            if (Array.isArray(value)) {{
                return value.join(', ');
            }}
            if (typeof value === 'object' && value !== null) {{
                return JSON.stringify(value, null, 2);
            }}
            return String(value);
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        // Update visualization when filters change
        setInterval(() => {{
            updateStats();
        }}, 100);
    </script>
</body>
</html>
    """


def _generate_field_color_mapping(field_types: List[str]) -> Dict[str, str]:
    """Generate consistent color mapping for field types."""
    colors = [
        "#FFE6E6",  # Light red
        "#E6F2FF",  # Light blue
        "#E6FFE6",  # Light green
        "#FFF0E6",  # Light orange
        "#F0E6FF",  # Light purple
        "#E6FFF0",  # Light mint
        "#FFE6F0",  # Light pink
        "#F0FFE6",  # Light lime
        "#FFFAE6",  # Light yellow
        "#E6F0FF",  # Light sky blue
        "#F0FFE6",  # Light lime green
        "#FFE6FA",  # Light magenta
    ]

    field_colors = {}
    for i, field_type in enumerate(sorted(field_types)):
        field_colors[field_type] = colors[i % len(colors)]

    return field_colors
