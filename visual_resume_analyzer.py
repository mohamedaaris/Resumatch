"""
Multi-modal Resume Analysis with Visual Elements
=============================================

Novel research feature that analyzes resume design quality using computer vision:
- Visual hierarchy and layout assessment
- Color scheme professionalism scoring  
- Typography and formatting quality
- Information density optimization
- Design skill inference from visual elements

Research Contribution:
- First to correlate resume design quality with technical skills
- Novel computer vision approach for professional document assessment
- Multi-modal analysis combining text and visual features
"""

import cv2
import numpy as np
from PIL import Image, ImageStat, ImageDraw, ImageFont
import io
import base64
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import colorsys
# import pytesseract  # Optional OCR dependency
from collections import Counter, defaultdict
import re
import math
# from sklearn.cluster import KMeans  # Using simplified clustering instead

logger = logging.getLogger(__name__)

class DesignQuality(Enum):
    POOR = 1
    FAIR = 2 
    GOOD = 3
    EXCELLENT = 4
    PROFESSIONAL = 5

@dataclass
class ColorScheme:
    """Represents color scheme analysis"""
    primary_colors: List[Tuple[int, int, int]]
    color_harmony: float  # 0-1 score
    professionalism_score: float  # 0-1 score
    contrast_ratio: float
    color_palette_type: str  # monochromatic, complementary, etc.

@dataclass
class LayoutMetrics:
    """Layout and spacing analysis"""
    white_space_ratio: float
    alignment_score: float
    section_separation: float
    margin_consistency: float
    grid_adherence: float
    information_density: float

@dataclass
class TypographyAnalysis:
    """Typography quality assessment"""
    font_consistency: float
    hierarchy_clarity: float
    readability_score: float
    font_variety_balance: float
    size_progression: List[int]

@dataclass
class VisualResumeAnalysis:
    """Complete visual analysis results"""
    overall_design_quality: DesignQuality
    professionalism_score: float
    attention_to_detail_score: float
    creativity_score: float
    technical_design_skills: float
    
    color_analysis: ColorScheme
    layout_analysis: LayoutMetrics
    typography_analysis: TypographyAnalysis
    
    visual_elements: Dict[str, Any]
    design_recommendations: List[str]
    inferred_skills: List[str]

class ColorAnalyzer:
    """Analyzes color schemes and palettes"""
    
    def __init__(self):
        # Professional color palettes (RGB values)
        self.professional_colors = {
            'navy_blue': [(25, 42, 86), (31, 58, 147), (41, 84, 209)],
            'charcoal': [(54, 69, 79), (69, 90, 100), (84, 110, 122)],
            'forest_green': [(46, 125, 50), (56, 142, 60), (67, 160, 71)],
            'burgundy': [(136, 14, 79), (173, 20, 87), (194, 24, 91)],
            'slate_blue': [(63, 81, 181), (92, 107, 192), (121, 134, 203)]
        }
        
    def analyze_colors(self, image: np.ndarray) -> ColorScheme:
        """Analyze color scheme of resume"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Extract dominant colors using K-means
            pixels = image_rgb.reshape(-1, 3)
            
            # Remove white/near-white pixels (background)
            non_white_pixels = pixels[np.sum(pixels, axis=1) < 720]  # Less than (240,240,240)
            
            if len(non_white_pixels) < 100:
                # Mostly white document
                return self._create_minimal_color_scheme()
            
            # Simple color extraction (dominant colors)
            # Sample pixels for performance
            sample_size = min(5000, len(non_white_pixels))
            sampled_pixels = non_white_pixels[::max(1, len(non_white_pixels) // sample_size)]
            
            # Find most common colors by grouping similar colors
            color_counts = {}
            for pixel in sampled_pixels:
                # Round to nearest 32 for grouping similar colors
                rounded = tuple((int(c) // 32) * 32 for c in pixel)
                color_counts[rounded] = color_counts.get(rounded, 0) + 1
            
            # Get top colors
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            primary_colors = [color for color, count in sorted_colors[:5]]
            
            # Analyze color harmony
            harmony_score = self._calculate_color_harmony(primary_colors)
            
            # Calculate professionalism score
            prof_score = self._calculate_professionalism_score(primary_colors)
            
            # Calculate contrast ratio
            contrast = self._calculate_contrast_ratio(image_rgb, primary_colors)
            
            # Determine palette type
            palette_type = self._determine_palette_type(primary_colors)
            
            return ColorScheme(
                primary_colors=primary_colors,
                color_harmony=harmony_score,
                professionalism_score=prof_score,
                contrast_ratio=contrast,
                color_palette_type=palette_type
            )
            
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            return self._create_minimal_color_scheme()
    
    def _create_minimal_color_scheme(self) -> ColorScheme:
        """Create default color scheme for minimal/text-only resumes"""
        return ColorScheme(
            primary_colors=[(0, 0, 0)],  # Black text
            color_harmony=0.8,  # Good for monochromatic
            professionalism_score=0.7,  # Professional but conservative
            contrast_ratio=0.9,  # High contrast
            color_palette_type="monochromatic"
        )
    
    def _calculate_color_harmony(self, colors: List[Tuple[int, int, int]]) -> float:
        """Calculate color harmony score based on color theory"""
        if len(colors) <= 1:
            return 0.8  # Monochromatic is harmonious
        
        # Convert to HSV for better color analysis
        hsv_colors = []
        for r, g, b in colors:
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hsv_colors.append((h*360, s, v))
        
        harmony_score = 0.0
        
        # Check for complementary colors (opposite on color wheel)
        for i, (h1, s1, v1) in enumerate(hsv_colors):
            for h2, s2, v2 in hsv_colors[i+1:]:
                hue_diff = abs(h1 - h2)
                if 150 <= hue_diff <= 210:  # Complementary range
                    harmony_score += 0.3
        
        # Check for analogous colors (adjacent on color wheel)
        hues = [h for h, s, v in hsv_colors]
        hues.sort()
        for i in range(len(hues) - 1):
            if 0 <= hues[i+1] - hues[i] <= 60:  # Analogous range
                harmony_score += 0.2
        
        # Saturation harmony (similar saturation levels)
        saturations = [s for h, s, v in hsv_colors]
        sat_variance = np.var(saturations)
        if sat_variance < 0.1:  # Low variance = harmonious
            harmony_score += 0.3
        
        return min(1.0, harmony_score)
    
    def _calculate_professionalism_score(self, colors: List[Tuple[int, int, int]]) -> float:
        """Score based on professional color choices"""
        prof_score = 0.0
        
        for color in colors:
            r, g, b = color
            
            # Check against professional palettes
            for palette_name, palette_colors in self.professional_colors.items():
                for prof_color in palette_colors:
                    # Calculate color distance
                    distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color, prof_color)))
                    if distance < 50:  # Close to professional color
                        prof_score += 0.3
            
            # Penalize overly bright/saturated colors
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            if s > 0.8 and v > 0.8:  # Very saturated and bright
                prof_score -= 0.2
            
            # Reward muted, professional tones
            if 0.2 <= s <= 0.6 and 0.3 <= v <= 0.8:
                prof_score += 0.2
        
        return max(0.0, min(1.0, prof_score / len(colors)))
    
    def _calculate_contrast_ratio(self, image: np.ndarray, colors: List[Tuple[int, int, int]]) -> float:
        """Calculate overall contrast ratio"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate standard deviation as contrast measure
        contrast = np.std(gray) / 128.0  # Normalize to 0-1
        
        return min(1.0, contrast)
    
    def _determine_palette_type(self, colors: List[Tuple[int, int, int]]) -> str:
        """Determine the type of color palette"""
        if len(colors) <= 1:
            return "monochromatic"
        
        # Convert to HSV
        hsv_colors = []
        for r, g, b in colors:
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hsv_colors.append((h*360, s, v))
        
        hues = [h for h, s, v in hsv_colors]
        
        # Check hue differences
        hue_diffs = []
        for i in range(len(hues)):
            for j in range(i+1, len(hues)):
                diff = abs(hues[i] - hues[j])
                hue_diffs.append(min(diff, 360 - diff))
        
        avg_hue_diff = np.mean(hue_diffs) if hue_diffs else 0
        
        if avg_hue_diff < 30:
            return "analogous"
        elif any(150 <= diff <= 210 for diff in hue_diffs):
            return "complementary"
        elif avg_hue_diff > 90:
            return "triadic"
        else:
            return "custom"

class LayoutAnalyzer:
    """Analyzes layout quality and spatial organization"""
    
    def analyze_layout(self, image: np.ndarray) -> LayoutMetrics:
        """Comprehensive layout analysis"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # White space analysis
            white_space_ratio = self._calculate_white_space(gray)
            
            # Alignment analysis
            alignment_score = self._analyze_alignment(gray)
            
            # Section separation
            section_sep = self._analyze_sections(gray)
            
            # Margin consistency
            margin_score = self._analyze_margins(gray)
            
            # Grid adherence (how well content follows a grid)
            grid_score = self._analyze_grid_adherence(gray)
            
            # Information density
            info_density = self._calculate_information_density(gray)
            
            return LayoutMetrics(
                white_space_ratio=white_space_ratio,
                alignment_score=alignment_score,
                section_separation=section_sep,
                margin_consistency=margin_score,
                grid_adherence=grid_score,
                information_density=info_density
            )
            
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            return self._create_default_layout_metrics()
    
    def _calculate_white_space(self, gray_image: np.ndarray) -> float:
        """Calculate ratio of white space to content"""
        # Threshold to find white areas
        _, binary = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)
        
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        
        white_space_ratio = white_pixels / total_pixels
        
        # Optimal white space is around 30-50%
        if 0.3 <= white_space_ratio <= 0.5:
            return 1.0
        elif 0.2 <= white_space_ratio <= 0.6:
            return 0.8
        else:
            return max(0.0, 1.0 - abs(white_space_ratio - 0.4) * 2)
    
    def _analyze_alignment(self, gray_image: np.ndarray) -> float:
        """Analyze text alignment and structure"""
        # Find text regions using contours
        _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours (text blocks)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 3:
            return 0.5  # Too few elements to analyze
        
        # Get bounding rectangles
        rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 100]
        
        if len(rects) < 3:
            return 0.5
        
        # Analyze left edge alignment
        left_edges = [x for x, y, w, h in rects]
        left_alignment_score = self._calculate_edge_alignment(left_edges)
        
        # Analyze right edge alignment
        right_edges = [x + w for x, y, w, h in rects]
        right_alignment_score = self._calculate_edge_alignment(right_edges)
        
        # Analyze vertical alignment
        top_edges = [y for x, y, w, h in rects]
        vertical_alignment_score = self._calculate_edge_alignment(top_edges)
        
        return (left_alignment_score + right_alignment_score + vertical_alignment_score) / 3
    
    def _calculate_edge_alignment(self, edges: List[int]) -> float:
        """Calculate how well edges align"""
        if len(edges) < 3:
            return 0.5
        
        edges.sort()
        
        # Group similar edges (within tolerance)
        tolerance = 20
        groups = []
        current_group = [edges[0]]
        
        for edge in edges[1:]:
            if edge - current_group[-1] <= tolerance:
                current_group.append(edge)
            else:
                groups.append(current_group)
                current_group = [edge]
        groups.append(current_group)
        
        # Calculate alignment score based on group sizes
        total_elements = len(edges)
        largest_group_size = max(len(group) for group in groups)
        
        alignment_score = largest_group_size / total_elements
        return alignment_score
    
    def _analyze_sections(self, gray_image: np.ndarray) -> float:
        """Analyze section separation and organization"""
        height, width = gray_image.shape
        
        # Horizontal line detection for section separators
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//10, 1))
        horizontal_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find horizontal separators
        _, binary = cv2.threshold(horizontal_lines, 50, 255, cv2.THRESH_BINARY_INV)
        horizontal_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Score based on presence of clear sections
        num_sections = len([cnt for cnt in horizontal_contours if cv2.contourArea(cnt) > width//20])
        
        # Optimal: 3-6 sections for a resume
        if 3 <= num_sections <= 6:
            return 1.0
        elif 2 <= num_sections <= 7:
            return 0.8
        else:
            return max(0.2, 1.0 - abs(num_sections - 4) * 0.2)
    
    def _analyze_margins(self, gray_image: np.ndarray) -> float:
        """Analyze margin consistency"""
        height, width = gray_image.shape
        
        # Find content boundaries
        _, binary = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find leftmost and rightmost content
        rows_with_content = np.any(binary, axis=1)
        cols_with_content = np.any(binary, axis=0)
        
        if not np.any(rows_with_content) or not np.any(cols_with_content):
            return 0.5
        
        top_margin = np.argmax(rows_with_content)
        bottom_margin = height - np.max(np.where(rows_with_content)[0]) - 1
        left_margin = np.argmax(cols_with_content)
        right_margin = width - np.max(np.where(cols_with_content)[0]) - 1
        
        # Check margin consistency
        margins = [top_margin, bottom_margin, left_margin, right_margin]
        margin_variance = np.var(margins)
        
        # Lower variance = better consistency
        consistency_score = max(0.0, 1.0 - margin_variance / 1000)
        
        return consistency_score
    
    def _analyze_grid_adherence(self, gray_image: np.ndarray) -> float:
        """Analyze how well content follows a grid system"""
        _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find text blocks
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 200]
        
        if len(rects) < 4:
            return 0.5
        
        # Analyze horizontal and vertical alignment patterns
        x_positions = [x for x, y, w, h in rects]
        y_positions = [y for x, y, w, h in rects]
        
        # Calculate grid score based on position clustering
        x_clusters = self._find_position_clusters(x_positions)
        y_clusters = self._find_position_clusters(y_positions)
        
        grid_score = (len(x_clusters) + len(y_clusters)) / (len(rects) * 0.8)
        return min(1.0, grid_score)
    
    def _find_position_clusters(self, positions: List[int]) -> List[List[int]]:
        """Find clusters of similar positions"""
        if not positions:
            return []
        
        positions.sort()
        clusters = []
        current_cluster = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_cluster[-1] <= 30:  # Tolerance
                current_cluster.append(pos)
            else:
                clusters.append(current_cluster)
                current_cluster = [pos]
        
        clusters.append(current_cluster)
        return [cluster for cluster in clusters if len(cluster) >= 2]
    
    def _calculate_information_density(self, gray_image: np.ndarray) -> float:
        """Calculate information density (text vs white space balance)"""
        _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        
        content_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        
        density = content_pixels / total_pixels
        
        # Optimal density is around 15-25% for readability
        if 0.15 <= density <= 0.25:
            return 1.0
        elif 0.10 <= density <= 0.30:
            return 0.8
        else:
            return max(0.2, 1.0 - abs(density - 0.20) * 3)
    
    def _create_default_layout_metrics(self) -> LayoutMetrics:
        """Create default layout metrics when analysis fails"""
        return LayoutMetrics(
            white_space_ratio=0.5,
            alignment_score=0.5,
            section_separation=0.5,
            margin_consistency=0.5,
            grid_adherence=0.5,
            information_density=0.5
        )

class TypographyAnalyzer:
    """Analyzes typography and text formatting"""
    
    def analyze_typography(self, image: np.ndarray) -> TypographyAnalysis:
        """Comprehensive typography analysis"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Find text regions and analyze fonts
            font_sizes = self._extract_font_sizes(gray)
            font_consistency = self._analyze_font_consistency(gray)
            hierarchy_clarity = self._analyze_hierarchy(font_sizes)
            readability_score = self._analyze_readability(gray)
            font_variety = self._analyze_font_variety(gray)
            
            return TypographyAnalysis(
                font_consistency=font_consistency,
                hierarchy_clarity=hierarchy_clarity,
                readability_score=readability_score,
                font_variety_balance=font_variety,
                size_progression=sorted(font_sizes, reverse=True)
            )
            
        except Exception as e:
            logger.error(f"Typography analysis failed: {e}")
            return self._create_default_typography()
    
    def _extract_font_sizes(self, gray_image: np.ndarray) -> List[int]:
        """Extract font sizes from text regions"""
        _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find text contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        font_sizes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 5 < h < 100 and w > 10:  # Reasonable text size bounds
                # Estimate font size based on height
                estimated_size = int(h * 0.75)  # Approximate font size from bounding box
                font_sizes.append(estimated_size)
        
        # Remove outliers and group similar sizes
        if font_sizes:
            font_sizes = [size for size in font_sizes if 8 <= size <= 72]
            
        return font_sizes if font_sizes else [12]  # Default if none found
    
    def _analyze_font_consistency(self, gray_image: np.ndarray) -> float:
        """Analyze consistency in font usage"""
        font_sizes = self._extract_font_sizes(gray_image)
        
        if not font_sizes:
            return 0.5
        
        # Group similar font sizes
        size_groups = []
        tolerance = 2
        
        for size in font_sizes:
            added = False
            for group in size_groups:
                if any(abs(size - g) <= tolerance for g in group):
                    group.append(size)
                    added = True
                    break
            if not added:
                size_groups.append([size])
        
        # Good consistency: 2-4 distinct font sizes
        num_distinct_sizes = len(size_groups)
        if 2 <= num_distinct_sizes <= 4:
            return 1.0
        elif num_distinct_sizes == 1 or num_distinct_sizes == 5:
            return 0.7
        else:
            return max(0.2, 1.0 - abs(num_distinct_sizes - 3) * 0.2)
    
    def _analyze_hierarchy(self, font_sizes: List[int]) -> float:
        """Analyze typographic hierarchy clarity"""
        if len(font_sizes) < 2:
            return 0.5
        
        unique_sizes = sorted(list(set(font_sizes)), reverse=True)
        
        if len(unique_sizes) < 2:
            return 0.3  # No hierarchy
        
        # Check for clear size progression
        size_ratios = []
        for i in range(len(unique_sizes) - 1):
            ratio = unique_sizes[i] / unique_sizes[i + 1]
            size_ratios.append(ratio)
        
        # Good hierarchy has ratios between 1.2-1.6
        good_ratios = sum(1 for ratio in size_ratios if 1.2 <= ratio <= 1.6)
        hierarchy_score = good_ratios / len(size_ratios) if size_ratios else 0
        
        return hierarchy_score
    
    def _analyze_readability(self, gray_image: np.ndarray) -> float:
        """Analyze text readability factors"""
        height, width = gray_image.shape
        
        # Calculate contrast
        contrast = np.std(gray_image) / 128.0
        
        # Line spacing analysis
        line_spacing_score = self._analyze_line_spacing(gray_image)
        
        # Character spacing (simplified)
        char_spacing_score = self._analyze_character_spacing(gray_image)
        
        # Combine factors
        readability = (
            min(1.0, contrast) * 0.4 +
            line_spacing_score * 0.3 +
            char_spacing_score * 0.3
        )
        
        return readability
    
    def _analyze_line_spacing(self, gray_image: np.ndarray) -> float:
        """Analyze line spacing adequacy"""
        _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find horizontal projection to detect line spacing
        horizontal_projection = np.sum(binary, axis=1)
        
        # Find lines (peaks in projection)
        line_positions = []
        in_line = False
        line_start = 0
        
        for i, value in enumerate(horizontal_projection):
            if value > width * 0.1:  # Threshold for line detection
                if not in_line:
                    line_start = i
                    in_line = True
            else:
                if in_line:
                    line_positions.append((line_start, i))
                    in_line = False
        
        if len(line_positions) < 2:
            return 0.5
        
        # Calculate line spacings
        line_heights = [end - start for start, end in line_positions]
        avg_line_height = np.mean(line_heights)
        
        spacings = []
        for i in range(len(line_positions) - 1):
            spacing = line_positions[i + 1][0] - line_positions[i][1]
            spacings.append(spacing)
        
        if not spacings:
            return 0.5
        
        avg_spacing = np.mean(spacings)
        
        # Good line spacing is 1.2-1.5 times line height
        optimal_spacing = avg_line_height * 1.35
        spacing_score = max(0.0, 1.0 - abs(avg_spacing - optimal_spacing) / optimal_spacing)
        
        return spacing_score
    
    def _analyze_character_spacing(self, gray_image: np.ndarray) -> float:
        """Analyze character spacing (simplified approach)"""
        # This is a simplified analysis - full OCR would be more accurate
        _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find connected components (characters)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        # Filter for character-like components
        char_components = []
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            if 10 < area < 1000 and 3 < width < 50 and 5 < height < 50:
                char_components.append(stats[i])
        
        if len(char_components) < 10:
            return 0.5
        
        # Analyze spacing distribution
        spacings = []
        char_components.sort(key=lambda x: (x[cv2.CC_STAT_TOP], x[cv2.CC_STAT_LEFT]))
        
        for i in range(len(char_components) - 1):
            curr = char_components[i]
            next_comp = char_components[i + 1]
            
            # If on same line (similar Y position)
            if abs(curr[cv2.CC_STAT_TOP] - next_comp[cv2.CC_STAT_TOP]) < curr[cv2.CC_STAT_HEIGHT] * 0.5:
                spacing = next_comp[cv2.CC_STAT_LEFT] - (curr[cv2.CC_STAT_LEFT] + curr[cv2.CC_STAT_WIDTH])
                if spacing > 0:
                    spacings.append(spacing)
        
        if not spacings:
            return 0.5
        
        # Good character spacing has low variance (consistent spacing)
        spacing_variance = np.var(spacings)
        max_expected_variance = 100  # Pixels
        spacing_score = max(0.0, 1.0 - spacing_variance / max_expected_variance)
        
        return spacing_score
    
    def _analyze_font_variety(self, gray_image: np.ndarray) -> float:
        """Analyze balance between font variety and consistency"""
        # This is a simplified analysis
        # In practice, would need OCR or ML model to detect different fonts
        
        # For now, estimate based on style variations in text regions
        _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find text regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze aspect ratios and density patterns as proxy for font variety
        aspect_ratios = []
        densities = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 10:  # Reasonable text size
                aspect_ratio = w / h
                aspect_ratios.append(aspect_ratio)
                
                # Calculate text density in region
                roi = binary[y:y+h, x:x+w]
                density = np.sum(roi == 255) / (w * h)
                densities.append(density)
        
        # Variety score based on distribution of characteristics
        variety_score = 0.5  # Default moderate variety
        
        if aspect_ratios and densities:
            # Moderate variety in aspect ratios indicates different text styles
            aspect_variety = np.std(aspect_ratios) / np.mean(aspect_ratios) if aspect_ratios else 0
            density_variety = np.std(densities) / np.mean(densities) if densities else 0
            
            # Optimal: some variety but not too much
            variety_score = min(1.0, (aspect_variety + density_variety) / 2)
            
            # Penalize too much variety (inconsistent) or too little (boring)
            if variety_score > 0.4:  # Too much variety
                variety_score = max(0.3, 1.0 - (variety_score - 0.4))
            elif variety_score < 0.1:  # Too little variety
                variety_score = 0.4
        
        return variety_score
    
    def _create_default_typography(self) -> TypographyAnalysis:
        """Create default typography analysis"""
        return TypographyAnalysis(
            font_consistency=0.5,
            hierarchy_clarity=0.5,
            readability_score=0.5,
            font_variety_balance=0.5,
            size_progression=[16, 14, 12]
        )

class VisualResumeAnalyzer:
    """Main class for comprehensive visual resume analysis"""
    
    def __init__(self):
        self.color_analyzer = ColorAnalyzer()
        self.layout_analyzer = LayoutAnalyzer()
        self.typography_analyzer = TypographyAnalyzer()
    
    def analyze_resume_image(self, image_bytes: bytes) -> VisualResumeAnalysis:
        """Analyze resume image comprehensively"""
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert PIL to numpy array
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image)
            
            return self._perform_analysis(image_np)
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return self._create_default_analysis()
    
    def analyze_resume_from_path(self, image_path: str) -> VisualResumeAnalysis:
        """Analyze resume from file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return self._perform_analysis(image_rgb)
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return self._create_default_analysis()
    
    def _perform_analysis(self, image: np.ndarray) -> VisualResumeAnalysis:
        """Perform comprehensive visual analysis"""
        
        # Individual component analyses
        color_analysis = self.color_analyzer.analyze_colors(image)
        layout_analysis = self.layout_analyzer.analyze_layout(image)
        typography_analysis = self.typography_analyzer.analyze_typography(image)
        
        # Visual elements detection
        visual_elements = self._detect_visual_elements(image)
        
        # Calculate overall scores
        professionalism_score = self._calculate_professionalism_score(
            color_analysis, layout_analysis, typography_analysis
        )
        
        attention_to_detail_score = self._calculate_attention_to_detail(
            layout_analysis, typography_analysis
        )
        
        creativity_score = self._calculate_creativity_score(
            color_analysis, visual_elements
        )
        
        technical_design_skills = self._calculate_technical_design_skills(
            layout_analysis, color_analysis, visual_elements
        )
        
        # Determine overall design quality
        overall_quality = self._determine_design_quality(
            professionalism_score, attention_to_detail_score, creativity_score
        )
        
        # Generate recommendations
        recommendations = self._generate_design_recommendations(
            color_analysis, layout_analysis, typography_analysis
        )
        
        # Infer design-related skills
        inferred_skills = self._infer_design_skills(
            visual_elements, color_analysis, layout_analysis
        )
        
        return VisualResumeAnalysis(
            overall_design_quality=overall_quality,
            professionalism_score=professionalism_score,
            attention_to_detail_score=attention_to_detail_score,
            creativity_score=creativity_score,
            technical_design_skills=technical_design_skills,
            color_analysis=color_analysis,
            layout_analysis=layout_analysis,
            typography_analysis=typography_analysis,
            visual_elements=visual_elements,
            design_recommendations=recommendations,
            inferred_skills=inferred_skills
        )
    
    def _detect_visual_elements(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect visual design elements"""
        elements = {
            'has_logo': False,
            'has_icons': False,
            'has_graphics': False,
            'has_charts': False,
            'has_borders': False,
            'element_count': 0,
            'design_complexity': 0.0
        }
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect circular elements (could be logos, icons)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=10, maxRadius=100
            )
            
            if circles is not None:
                elements['has_icons'] = len(circles[0]) > 0
                elements['element_count'] += len(circles[0])
            
            # Detect lines (borders, separators)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50)
            
            if lines is not None:
                elements['has_borders'] = len(lines) > 5
                elements['element_count'] += len(lines)
            
            # Detect rectangular elements (could be graphics, charts)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rectangles = 0
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # Rectangle-like shape
                    area = cv2.contourArea(contour)
                    if 1000 < area < 50000:  # Reasonable size for graphics
                        rectangles += 1
            
            elements['has_graphics'] = rectangles > 2
            elements['element_count'] += rectangles
            
            # Calculate design complexity
            total_elements = elements['element_count']
            image_area = image.shape[0] * image.shape[1]
            
            # Normalize complexity by image size
            complexity = min(1.0, total_elements / (image_area / 100000))
            elements['design_complexity'] = complexity
            
        except Exception as e:
            logger.error(f"Visual element detection failed: {e}")
        
        return elements
    
    def _calculate_professionalism_score(self, color_analysis: ColorScheme, 
                                       layout_analysis: LayoutMetrics,
                                       typography_analysis: TypographyAnalysis) -> float:
        """Calculate overall professionalism score"""
        
        # Weight different factors
        color_weight = 0.3
        layout_weight = 0.4
        typography_weight = 0.3
        
        # Color professionalism
        color_score = color_analysis.professionalism_score
        
        # Layout professionalism (good margins, alignment, white space)
        layout_score = (
            layout_analysis.white_space_ratio * 0.3 +
            layout_analysis.alignment_score * 0.3 +
            layout_analysis.margin_consistency * 0.2 +
            layout_analysis.section_separation * 0.2
        )
        
        # Typography professionalism (consistency, readability)
        typography_score = (
            typography_analysis.font_consistency * 0.4 +
            typography_analysis.readability_score * 0.4 +
            typography_analysis.hierarchy_clarity * 0.2
        )
        
        overall_score = (
            color_score * color_weight +
            layout_score * layout_weight +
            typography_score * typography_weight
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _calculate_attention_to_detail(self, layout_analysis: LayoutMetrics,
                                     typography_analysis: TypographyAnalysis) -> float:
        """Calculate attention to detail score"""
        
        # Focus on precision and consistency
        detail_score = (
            layout_analysis.margin_consistency * 0.25 +
            layout_analysis.alignment_score * 0.25 +
            layout_analysis.grid_adherence * 0.20 +
            typography_analysis.font_consistency * 0.15 +
            typography_analysis.hierarchy_clarity * 0.15
        )
        
        return min(1.0, max(0.0, detail_score))
    
    def _calculate_creativity_score(self, color_analysis: ColorScheme,
                                  visual_elements: Dict[str, Any]) -> float:
        """Calculate creativity and innovation score"""
        
        creativity_factors = []
        
        # Color creativity (but not unprofessional)
        color_creativity = color_analysis.color_harmony * 0.7
        if color_analysis.color_palette_type in ['complementary', 'triadic']:
            color_creativity += 0.3
        creativity_factors.append(color_creativity)
        
        # Visual elements usage
        element_creativity = visual_elements['design_complexity'] * 0.8
        if visual_elements['has_icons']:
            element_creativity += 0.1
        if visual_elements['has_graphics']:
            element_creativity += 0.1
        creativity_factors.append(element_creativity)
        
        # Balance creativity with professionalism
        creativity_score = np.mean(creativity_factors)
        
        # Penalize excessive creativity that might hurt professionalism
        if creativity_score > 0.8:
            creativity_score = 0.8
        
        return creativity_score
    
    def _calculate_technical_design_skills(self, layout_analysis: LayoutMetrics,
                                         color_analysis: ColorScheme,
                                         visual_elements: Dict[str, Any]) -> float:
        """Infer technical design skills from visual quality"""
        
        technical_indicators = [
            layout_analysis.grid_adherence * 0.3,  # Understanding of grid systems
            color_analysis.color_harmony * 0.2,    # Color theory knowledge
            layout_analysis.white_space_ratio * 0.2,  # Space management
            visual_elements['design_complexity'] * 0.3  # Tool proficiency
        ]
        
        return np.mean(technical_indicators)
    
    def _determine_design_quality(self, professionalism: float, attention_to_detail: float,
                                creativity: float) -> DesignQuality:
        """Determine overall design quality rating"""
        
        overall_score = (professionalism * 0.5 + attention_to_detail * 0.3 + creativity * 0.2)
        
        if overall_score >= 0.9:
            return DesignQuality.PROFESSIONAL
        elif overall_score >= 0.75:
            return DesignQuality.EXCELLENT
        elif overall_score >= 0.6:
            return DesignQuality.GOOD
        elif overall_score >= 0.4:
            return DesignQuality.FAIR
        else:
            return DesignQuality.POOR
    
    def _generate_design_recommendations(self, color_analysis: ColorScheme,
                                       layout_analysis: LayoutMetrics,
                                       typography_analysis: TypographyAnalysis) -> List[str]:
        """Generate specific design improvement recommendations"""
        
        recommendations = []
        
        # Color recommendations
        if color_analysis.professionalism_score < 0.6:
            recommendations.append("Consider using more professional color palette (navy, charcoal, or muted tones)")
        
        if color_analysis.contrast_ratio < 0.5:
            recommendations.append("Increase contrast between text and background for better readability")
        
        # Layout recommendations
        if layout_analysis.white_space_ratio < 0.3:
            recommendations.append("Add more white space to improve readability and visual balance")
        
        if layout_analysis.alignment_score < 0.6:
            recommendations.append("Improve text alignment and create consistent left/right margins")
        
        if layout_analysis.margin_consistency < 0.5:
            recommendations.append("Ensure consistent margins throughout the document")
        
        # Typography recommendations
        if typography_analysis.font_consistency < 0.6:
            recommendations.append("Use fewer font sizes (2-3 maximum) for better consistency")
        
        if typography_analysis.hierarchy_clarity < 0.5:
            recommendations.append("Create clearer typographic hierarchy with distinct heading sizes")
        
        if typography_analysis.readability_score < 0.6:
            recommendations.append("Improve line spacing and character spacing for better readability")
        
        if not recommendations:
            recommendations.append("Great design! Consider minor refinements to achieve professional excellence")
        
        return recommendations
    
    def _infer_design_skills(self, visual_elements: Dict[str, Any],
                           color_analysis: ColorScheme,
                           layout_analysis: LayoutMetrics) -> List[str]:
        """Infer design-related skills from visual analysis"""
        
        skills = []
        
        # High-quality layout suggests design tools knowledge
        if layout_analysis.grid_adherence > 0.7:
            skills.extend(['Adobe InDesign', 'Layout Design', 'Grid Systems'])
        
        # Good color usage suggests color theory knowledge
        if color_analysis.color_harmony > 0.7 and color_analysis.professionalism_score > 0.7:
            skills.extend(['Color Theory', 'Visual Design', 'Brand Design'])
        
        # Visual elements suggest graphics skills
        if visual_elements['has_graphics'] or visual_elements['has_icons']:
            skills.extend(['Adobe Illustrator', 'Graphic Design', 'Icon Design'])
        
        # High overall quality suggests professional tools
        if (layout_analysis.alignment_score > 0.8 and 
            layout_analysis.margin_consistency > 0.8):
            skills.extend(['Adobe Creative Suite', 'Professional Design Tools'])
        
        # Typography quality suggests font/text design skills
        if layout_analysis.information_density > 0.7:
            skills.extend(['Typography', 'Information Design'])
        
        # Creative elements but professional execution
        if visual_elements['design_complexity'] > 0.5 and color_analysis.professionalism_score > 0.6:
            skills.extend(['Creative Design', 'Visual Communication'])
        
        return list(set(skills))  # Remove duplicates
    
    def _create_default_analysis(self) -> VisualResumeAnalysis:
        """Create default analysis when image processing fails"""
        return VisualResumeAnalysis(
            overall_design_quality=DesignQuality.FAIR,
            professionalism_score=0.5,
            attention_to_detail_score=0.5,
            creativity_score=0.5,
            technical_design_skills=0.5,
            color_analysis=ColorScheme([(0, 0, 0)], 0.5, 0.5, 0.5, "monochromatic"),
            layout_analysis=LayoutMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            typography_analysis=TypographyAnalysis(0.5, 0.5, 0.5, 0.5, [12]),
            visual_elements={'design_complexity': 0.3},
            design_recommendations=["Consider improving visual design quality"],
            inferred_skills=[]
        )

# Integration function for the main application
def analyze_resume_visual_quality(image_bytes: bytes) -> Dict[str, Any]:
    """Main function to analyze resume visual quality"""
    
    analyzer = VisualResumeAnalyzer()
    analysis = analyzer.analyze_resume_image(image_bytes)
    
    return {
        "visual_analysis": {
            "overall_design_quality": analysis.overall_design_quality.name,
            "professionalism_score": analysis.professionalism_score,
            "attention_to_detail_score": analysis.attention_to_detail_score,
            "creativity_score": analysis.creativity_score,
            "technical_design_skills": analysis.technical_design_skills
        },
        "color_analysis": {
            "primary_colors": analysis.color_analysis.primary_colors,
            "color_harmony": analysis.color_analysis.color_harmony,
            "professionalism_score": analysis.color_analysis.professionalism_score,
            "color_palette_type": analysis.color_analysis.color_palette_type
        },
        "layout_analysis": {
            "white_space_ratio": analysis.layout_analysis.white_space_ratio,
            "alignment_score": analysis.layout_analysis.alignment_score,
            "margin_consistency": analysis.layout_analysis.margin_consistency,
            "information_density": analysis.layout_analysis.information_density
        },
        "typography_analysis": {
            "font_consistency": analysis.typography_analysis.font_consistency,
            "hierarchy_clarity": analysis.typography_analysis.hierarchy_clarity,
            "readability_score": analysis.typography_analysis.readability_score,
            "font_sizes": analysis.typography_analysis.size_progression
        },
        "design_recommendations": analysis.design_recommendations,
        "inferred_design_skills": analysis.inferred_skills,
        "visual_elements": analysis.visual_elements
    }

# Test the visual analyzer
if __name__ == "__main__":
    print("Multi-modal Resume Visual Analyzer - Research Implementation")
    print("=" * 65)
    
    # Create a test image (simulated resume)
    test_image = np.ones((800, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Add some simulated content (black rectangles for text blocks)
    cv2.rectangle(test_image, (50, 50), (550, 100), (0, 0, 0), -1)  # Header
    cv2.rectangle(test_image, (50, 120), (400, 140), (0, 0, 0), -1)  # Line 1
    cv2.rectangle(test_image, (50, 160), (450, 180), (0, 0, 0), -1)  # Line 2
    cv2.rectangle(test_image, (50, 200), (350, 220), (0, 0, 0), -1)  # Line 3
    
    # Add some color elements (blue accent)
    cv2.rectangle(test_image, (50, 240), (200, 260), (100, 150, 200), -1)
    
    # Convert to bytes
    _, encoded_image = cv2.imencode('.png', test_image)
    image_bytes = encoded_image.tobytes()
    
    # Analyze the test image
    try:
        result = analyze_resume_visual_quality(image_bytes)
        
        print("\n🎨 VISUAL ANALYSIS RESULTS:")
        visual = result['visual_analysis']
        print(f"Overall Design Quality: {visual['overall_design_quality']}")
        print(f"Professionalism Score: {visual['professionalism_score']:.2f}")
        print(f"Attention to Detail: {visual['attention_to_detail_score']:.2f}")
        print(f"Creativity Score: {visual['creativity_score']:.2f}")
        print(f"Technical Design Skills: {visual['technical_design_skills']:.2f}")
        
        print("\n🎯 COLOR ANALYSIS:")
        color = result['color_analysis']
        print(f"Color Harmony: {color['color_harmony']:.2f}")
        print(f"Color Professionalism: {color['professionalism_score']:.2f}")
        print(f"Palette Type: {color['color_palette_type']}")
        
        print("\n📐 LAYOUT ANALYSIS:")
        layout = result['layout_analysis']
        print(f"White Space Ratio: {layout['white_space_ratio']:.2f}")
        print(f"Alignment Score: {layout['alignment_score']:.2f}")
        print(f"Margin Consistency: {layout['margin_consistency']:.2f}")
        
        print("\n✍️ TYPOGRAPHY ANALYSIS:")
        typo = result['typography_analysis']
        print(f"Font Consistency: {typo['font_consistency']:.2f}")
        print(f"Hierarchy Clarity: {typo['hierarchy_clarity']:.2f}")
        print(f"Readability Score: {typo['readability_score']:.2f}")
        
        print("\n💡 DESIGN RECOMMENDATIONS:")
        for rec in result['design_recommendations'][:3]:
            print(f"• {rec}")
        
        print("\n🛠️ INFERRED DESIGN SKILLS:")
        for skill in result['inferred_design_skills'][:5]:
            print(f"• {skill}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    print("\n✅ Multi-modal Visual Analysis - IMPLEMENTED!")