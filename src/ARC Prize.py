"""
================================================================================
FROM PATTERN RECOGNITION TO REASONING: ARC PRIZE 2026 PAPER TRACK
================================================================================

A comprehensive framework for the Abstraction and Reasoning Corpus (ARC) Challenge.
This implementation explores the transition from pure pattern recognition to 
abstract reasoning capabilities in AI systems.

Author: ARC Research Team
Version: 1.0.0
Year: 2026
================================================================================
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from itertools import product, combinations, permutations
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: CORE DATA STRUCTURES
# =============================================================================

@dataclass
class ARCTask:
    """
    Represents a single ARC task with training and test examples.
    """
    task_id: str
    train_examples: List[Tuple[np.ndarray, np.ndarray]]
    test_examples: List[Tuple[np.ndarray, Optional[np.ndarray]]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate task structure after initialization."""
        if not self.train_examples:
            raise ValueError(f"Task {self.task_id} must have at least one training example")
    
    @property
    def num_train(self) -> int:
        return len(self.train_examples)
    
    @property
    def num_test(self) -> int:
        return len(self.test_examples)
    
    def get_input_shapes(self) -> List[Tuple[int, int]]:
        """Get all input grid shapes."""
        return [ex[0].shape for ex in self.train_examples]
    
    def get_output_shapes(self) -> List[Tuple[int, int]]:
        """Get all output grid shapes."""
        return [ex[1].shape for ex in self.train_examples if ex[1] is not None]


@dataclass
class ReasoningTrace:
    """
    Captures the reasoning process for solving an ARC task.
    """
    task_id: str
    detected_patterns: List[str] = field(default_factory=list)
    applied_transformations: List[str] = field(default_factory=list)
    hypothesis_scores: Dict[str, float] = field(default_factory=dict)
    reasoning_steps: List[str] = field(default_factory=list)
    confidence: float = 0.0
    success: bool = False
    
    def add_step(self, step: str, pattern: Optional[str] = None):
        """Add a reasoning step with optional pattern detection."""
        self.reasoning_steps.append(step)
        if pattern:
            self.detected_patterns.append(pattern)


class ARCColor:
    """
    ARC color palette with semantic meanings.
    """
    PALETTE = {
        0: ('black', 'background/empty'),
        1: ('blue', 'water/fill'),
        2: ('red', 'danger/marker'),
        3: ('green', 'growth/life'),
        4: ('yellow', 'attention/sun'),
        5: ('gray', 'neutral/structure'),
        6: ('magenta', 'special/target'),
        7: ('orange', 'warmth/highlight'),
        8: ('cyan', 'cool/secondary'),
        9: ('brown', 'earth/ground')
    }
    
    @classmethod
    def get_color(cls, value: int) -> str:
        return cls.PALETTE.get(value, ('unknown', 'unknown'))[0]
    
    @classmethod
    def get_semantic(cls, value: int) -> str:
        return cls.PALETTE.get(value, ('unknown', 'unknown'))[1]


# =============================================================================
# SECTION 2: PATTERN RECOGNITION MODULE
# =============================================================================

class PatternRecognizer:
    """
    Identifies visual and structural patterns in ARC grids.
    """
    
    def __init__(self):
        self.detected_patterns: Dict[str, Any] = {}
    
    def analyze(self, grid: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive pattern analysis of a grid.
        """
        patterns = {
            'basic_stats': self._basic_statistics(grid),
            'color_analysis': self._analyze_colors(grid),
            'shape_analysis': self._analyze_shapes(grid),
            'symmetry': self._detect_symmetry(grid),
            'repetition': self._detect_repetition(grid),
            'boundaries': self._detect_boundaries(grid),
            'connected_components': self._find_connected_components(grid),
            'spatial_relations': self._analyze_spatial_relations(grid)
        }
        self.detected_patterns = patterns
        return patterns
    
    def _basic_statistics(self, grid: np.ndarray) -> Dict[str, Any]:
        """Calculate basic grid statistics."""
        return {
            'shape': grid.shape,
            'total_cells': grid.size,
            'unique_colors': len(np.unique(grid)),
            'color_distribution': dict(Counter(grid.flatten())),
            'non_zero_ratio': np.count_nonzero(grid) / grid.size,
            'density': np.sum(grid > 0) / grid.size
        }
    
    def _analyze_colors(self, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze color usage patterns."""
        unique, counts = np.unique(grid, return_counts=True)
        color_info = {}
        
        for color, count in zip(unique, counts):
            positions = np.argwhere(grid == color)
            color_info[int(color)] = {
                'count': int(count),
                'positions': positions.tolist(),
                'centroid': positions.mean(axis=0).tolist() if len(positions) > 0 else None,
                'bounding_box': self._get_bounding_box(positions) if len(positions) > 0 else None
            }
        
        return color_info
    
    def _get_bounding_box(self, positions: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box (min_row, min_col, max_row, max_col)."""
        if len(positions) == 0:
            return (0, 0, 0, 0)
        return (
            int(positions[:, 0].min()),
            int(positions[:, 1].min()),
            int(positions[:, 0].max()),
            int(positions[:, 1].max())
        )
    
    def _analyze_shapes(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect and analyze shapes in the grid."""
        shapes = {}
        visited = np.zeros_like(grid, dtype=bool)
        shape_id = 0
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0 and not visited[i, j]:
                    shape_mask, visited = self._flood_fill(grid, i, j, visited)
                    shape_cells = np.argwhere(shape_mask)
                    shapes[f'shape_{shape_id}'] = {
                        'color': int(grid[i, j]),
                        'cells': shape_cells.tolist(),
                        'size': len(shape_cells),
                        'bbox': self._get_bounding_box(shape_cells),
                        'centroid': shape_cells.mean(axis=0).tolist()
                    }
                    shape_id += 1
        
        return shapes
    
    def _flood_fill(self, grid: np.ndarray, start_i: int, start_j: int, 
                    visited: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Flood fill to find connected component of same color."""
        color = grid[start_i, start_j]
        mask = np.zeros_like(grid, dtype=bool)
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            if (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and 
                not visited[i, j] and grid[i, j] == color):
                visited[i, j] = True
                mask[i, j] = True
                stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
        
        return mask, visited
    
    def _detect_symmetry(self, grid: np.ndarray) -> Dict[str, bool]:
        """Detect various symmetry patterns."""
        return {
            'horizontal': np.array_equal(grid, np.flip(grid, axis=0)),
            'vertical': np.array_equal(grid, np.flip(grid, axis=1)),
            'rotational_90': np.array_equal(grid, np.rot90(grid)),
            'rotational_180': np.array_equal(grid, np.rot90(grid, 2)),
            'rotational_270': np.array_equal(grid, np.rot90(grid, 3)),
            'diagonal': np.array_equal(grid, grid.T),
            'anti_diagonal': np.array_equal(grid, np.flip(np.flip(grid, 0).T, 0))
        }
    
    def _detect_repetition(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect repeating patterns."""
        repetition_info = {}
        
        # Check for row repetition
        rows = [tuple(row) for row in grid]
        row_patterns = {}
        for i, row in enumerate(rows):
            if row in row_patterns:
                row_patterns[row].append(i)
            else:
                row_patterns[row] = [i]
        
        repetition_info['repeating_rows'] = {
            str(k): v for k, v in row_patterns.items() if len(v) > 1
        }
        
        # Check for column repetition
        cols = [tuple(col) for col in grid.T]
        col_patterns = {}
        for i, col in enumerate(cols):
            if col in col_patterns:
                col_patterns[col].append(i)
            else:
                col_patterns[col] = [i]
        
        repetition_info['repeating_cols'] = {
            str(k): v for k, v in col_patterns.items() if len(v) > 1
        }
        
        # Detect periodic patterns
        repetition_info['periodicity'] = self._detect_periodicity(grid)
        
        return repetition_info
    
    def _detect_periodicity(self, grid: np.ndarray) -> Dict[str, Optional[int]]:
        """Detect periodic patterns in rows and columns."""
        def find_period(sequence):
            n = len(sequence)
            for p in range(1, n // 2 + 1):
                if n % p == 0:
                    is_periodic = all(sequence[i] == sequence[i % p] for i in range(n))
                    if is_periodic:
                        return p
            return None
        
        row_periods = [find_period(list(row)) for row in grid]
        col_periods = [find_period(list(col)) for col in grid.T]
        
        return {
            'row_period': max(set(row_periods), key=row_periods.count) if row_periods else None,
            'col_period': max(set(col_periods), key=col_periods.count) if col_periods else None
        }
    
    def _detect_boundaries(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect boundary patterns and enclosures."""
        boundaries = {
            'has_border': self._check_border(grid),
            'enclosed_regions': self._find_enclosed_regions(grid),
            'dividers': self._find_dividers(grid)
        }
        return boundaries
    
    def _check_border(self, grid: np.ndarray) -> Dict[str, Any]:
        """Check for border patterns."""
        if grid.shape[0] < 2 or grid.shape[1] < 2:
            return {'present': False}
        
        top = grid[0, :]
        bottom = grid[-1, :]
        left = grid[:, 0]
        right = grid[:, -1]
        
        return {
            'present': bool(
                np.all(top == top[0]) or 
                np.all(bottom == bottom[0]) or
                np.all(left == left[0]) or
                np.all(right == right[0])
            ),
            'top_color': int(top[0]) if np.all(top == top[0]) else None,
            'bottom_color': int(bottom[0]) if np.all(bottom == bottom[0]) else None,
            'left_color': int(left[0]) if np.all(left == left[0]) else None,
            'right_color': int(right[0]) if np.all(right == right[0]) else None
        }
    
    def _find_enclosed_regions(self, grid: np.ndarray) -> List[Dict]:
        """Find regions enclosed by non-zero cells."""
        # Simplified version - finds connected regions of zeros surrounded by non-zeros
        enclosed = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 0 and not visited[i, j]:
                    region, is_enclosed, visited = self._flood_fill_check_enclosure(
                        grid, i, j, visited
                    )
                    if is_enclosed:
                        enclosed.append({
                            'size': len(region),
                            'cells': region
                        })
        
        return enclosed
    
    def _flood_fill_check_enclosure(self, grid: np.ndarray, start_i: int, start_j: int,
                                    visited: np.ndarray) -> Tuple[List, bool, np.ndarray]:
        """Flood fill and check if region is enclosed."""
        region = []
        is_enclosed = True
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            if i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1]:
                is_enclosed = False
                continue
            if visited[i, j] or grid[i, j] != 0:
                continue
            
            visited[i, j] = True
            region.append((i, j))
            stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
        
        return region, is_enclosed, visited
    
    def _find_dividers(self, grid: np.ndarray) -> List[Dict]:
        """Find horizontal and vertical dividers."""
        dividers = []
        
        # Horizontal dividers
        for i in range(1, grid.shape[0] - 1):
            if np.all(grid[i, :] == grid[i, 0]) and grid[i, 0] != 0:
                dividers.append({
                    'type': 'horizontal',
                    'position': i,
                    'color': int(grid[i, 0])
                })
        
        # Vertical dividers
        for j in range(1, grid.shape[1] - 1):
            if np.all(grid[:, j] == grid[0, j]) and grid[0, j] != 0:
                dividers.append({
                    'type': 'vertical',
                    'position': j,
                    'color': int(grid[0, j])
                })
        
        return dividers
    
    def _find_connected_components(self, grid: np.ndarray) -> Dict[str, Any]:
        """Find all connected components by color."""
        components = defaultdict(list)
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j]:
                    color = grid[i, j]
                    mask, visited = self._flood_fill_by_color(grid, i, j, visited)
                    cells = np.argwhere(mask)
                    if len(cells) > 0:
                        components[int(color)].append({
                            'cells': cells.tolist(),
                            'size': len(cells),
                            'bbox': self._get_bounding_box(cells)
                        })
        
        return dict(components)
    
    def _flood_fill_by_color(self, grid: np.ndarray, start_i: int, start_j: int,
                             visited: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Flood fill regardless of color."""
        mask = np.zeros_like(grid, dtype=bool)
        stack = [(start_i, start_j)]
        start_color = grid[start_i, start_j]
        
        while stack:
            i, j = stack.pop()
            if (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and 
                not visited[i, j] and grid[i, j] == start_color):
                visited[i, j] = True
                mask[i, j] = True
                stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
        
        return mask, visited
    
    def _analyze_spatial_relations(self, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial relationships between objects."""
        shapes = self._analyze_shapes(grid)['shapes']
        relations = []
        
        shape_list = list(shapes.items())
        for i, (name1, s1) in enumerate(shape_list):
            for name2, s2 in shape_list[i+1:]:
                relation = self._compute_relation(s1, s2)
                relations.append({
                    'objects': (name1, name2),
                    'relation': relation
                })
        
        return {
            'pairwise_relations': relations,
            'spatial_summary': self._summarize_spatial(shapes)
        }
    
    def _compute_relation(self, shape1: Dict, shape2: Dict) -> Dict[str, Any]:
        """Compute spatial relation between two shapes."""
        c1 = np.array(shape1['centroid'])
        c2 = np.array(shape2['centroid'])
        diff = c2 - c1
        
        return {
            'distance': float(np.linalg.norm(diff)),
            'direction': 'right' if diff[1] > 0 else 'left' if diff[1] < 0 else 'same_col',
            'vertical': 'below' if diff[0] > 0 else 'above' if diff[0] < 0 else 'same_row',
            'size_ratio': shape1['size'] / shape2['size'] if shape2['size'] > 0 else float('inf')
        }
    
    def _summarize_spatial(self, shapes: Dict) -> Dict[str, Any]:
        """Summarize overall spatial arrangement."""
        if not shapes:
            return {}
        
        centroids = [np.array(s['centroid']) for s in shapes.values()]
        centroids = np.array(centroids)
        
        return {
            'num_objects': len(shapes),
            'centroid_spread': {
                'row_std': float(centroids[:, 0].std()),
                'col_std': float(centroids[:, 1].std())
            },
            'size_distribution': [s['size'] for s in shapes.values()]
        }


# =============================================================================
# SECTION 3: TRANSFORMATION ENGINE
# =============================================================================

class Transformation(ABC):
    """Base class for all ARC transformations."""
    
    @abstractmethod
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply the transformation to a grid."""
        pass
    
    @abstractmethod
    def is_applicable(self, grid: np.ndarray) -> bool:
        """Check if transformation is applicable."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return transformation name."""
        pass


class GeometricTransformation(Transformation):
    """Geometric transformations: rotation, reflection, scaling."""
    
    def __init__(self, operation: str, params: Dict = None):
        self.operation = operation
        self.params = params or {}
    
    @property
    def name(self) -> str:
        return f"geometric_{self.operation}"
    
    def is_applicable(self, grid: np.ndarray) -> bool:
        return True
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        ops = {
            'rotate_90': lambda g: np.rot90(g, 1),
            'rotate_180': lambda g: np.rot90(g, 2),
            'rotate_270': lambda g: np.rot90(g, 3),
            'flip_h': lambda g: np.flip(g, axis=0),
            'flip_v': lambda g: np.flip(g, axis=1),
            'transpose': lambda g: g.T,
            'flip_diag': lambda g: np.flip(np.flip(g, 0).T, 0),
        }
        
        if self.operation in ops:
            return ops[self.operation](grid)
        elif self.operation == 'scale':
            factor = self.params.get('factor', 2)
            return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)
        
        return grid


class ColorTransformation(Transformation):
    """Color-based transformations."""
    
    def __init__(self, operation: str, params: Dict = None):
        self.operation = operation
        self.params = params or {}
    
    @property
    def name(self) -> str:
        return f"color_{self.operation}"
    
    def is_applicable(self, grid: np.ndarray) -> bool:
        if self.operation == 'replace':
            return self.params.get('from_color') in grid
        return True
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        
        if self.operation == 'replace':
            from_c = self.params.get('from_color')
            to_c = self.params.get('to_color')
            result[result == from_c] = to_c
        
        elif self.operation == 'invert':
            unique = np.unique(grid)
            mapping = {u: unique[-(i+1)] for i, u in enumerate(unique)}
            for k, v in mapping.items():
                result[grid == k] = v
        
        elif self.operation == 'fill_bg':
            bg_color = self.params.get('bg_color', 0)
            fill_color = self.params.get('fill_color', 1)
            result[result == bg_color] = fill_color
        
        elif self.operation == 'map_colors':
            mapping = self.params.get('mapping', {})
            for k, v in mapping.items():
                result[grid == k] = v
        
        return result


class CropTransformation(Transformation):
    """Cropping and extraction transformations."""
    
    def __init__(self, operation: str, params: Dict = None):
        self.operation = operation
        self.params = params or {}
    
    @property
    def name(self) -> str:
        return f"crop_{self.operation}"
    
    def is_applicable(self, grid: np.ndarray) -> bool:
        return True
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        if self.operation == 'crop_content':
            return self._crop_to_content(grid)
        elif self.operation == 'crop_bbox':
            return self._crop_by_bbox(grid)
        elif self.operation == 'extract_region':
            return self._extract_region(grid)
        return grid
    
    def _crop_to_content(self, grid: np.ndarray) -> np.ndarray:
        """Crop to non-zero content."""
        rows = np.any(grid != 0, axis=1)
        cols = np.any(grid != 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return grid
        
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        
        return grid[row_indices[0]:row_indices[-1]+1, 
                   col_indices[0]:col_indices[-1]+1]
    
    def _crop_by_bbox(self, grid: np.ndarray) -> np.ndarray:
        """Crop by bounding box specified in params."""
        r1, c1, r2, c2 = self.params.get('bbox', (0, 0, grid.shape[0], grid.shape[1]))
        return grid[r1:r2, c1:c2]
    
    def _extract_region(self, grid: np.ndarray) -> np.ndarray:
        """Extract a specific region based on color."""
        color = self.params.get('color', 1)
        mask = grid == color
        
        if not np.any(mask):
            return grid
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        
        result = grid[row_indices[0]:row_indices[-1]+1,
                     col_indices[0]:col_indices[-1]+1]
        return result


class PatternTransformation(Transformation):
    """Pattern-based transformations: tiling, repeating, etc."""
    
    def __init__(self, operation: str, params: Dict = None):
        self.operation = operation
        self.params = params or {}
    
    @property
    def name(self) -> str:
        return f"pattern_{self.operation}"
    
    def is_applicable(self, grid: np.ndarray) -> bool:
        return True
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        if self.operation == 'tile':
            n_rows = self.params.get('rows', 2)
            n_cols = self.params.get('cols', 2)
            return np.tile(grid, (n_rows, n_cols))
        
        elif self.operation == 'repeat_pattern':
            return self._repeat_pattern(grid)
        
        elif self.operation == 'mirror':
            axis = self.params.get('axis', 'h')
            if axis == 'h':
                return np.concatenate([grid, np.flip(grid, axis=0)], axis=0)
            else:
                return np.concatenate([grid, np.flip(grid, axis=1)], axis=1)
        
        elif self.operation == 'extend':
            return self._extend_pattern(grid)
        
        return grid
    
    def _repeat_pattern(self, grid: np.ndarray) -> np.ndarray:
        """Repeat pattern with variations."""
        times = self.params.get('times', 2)
        direction = self.params.get('direction', 'horizontal')
        
        result = grid.copy()
        for _ in range(times - 1):
            if direction == 'horizontal':
                result = np.concatenate([result, grid], axis=1)
            else:
                result = np.concatenate([result, grid], axis=0)
        
        return result
    
    def _extend_pattern(self, grid: np.ndarray) -> np.ndarray:
        """Extend pattern by analyzing edges."""
        # Simple extension: mirror the edge
        extend_by = self.params.get('extend_by', 1)
        direction = self.params.get('direction', 'right')
        
        result = grid.copy()
        if direction == 'right':
            extension = grid[:, -extend_by:]
            result = np.concatenate([result, np.flip(extension, axis=1)], axis=1)
        elif direction == 'down':
            extension = grid[-extend_by:, :]
            result = np.concatenate([result, np.flip(extension, axis=0)], axis=0)
        
        return result


class TransformationEngine:
    """
    Engine for applying and discovering transformations.
    """
    
    def __init__(self):
        self.pattern_recognizer = PatternRecognizer()
        self.transformations = self._build_transformation_library()
    
    def _build_transformation_library(self) -> List[Transformation]:
        """Build library of available transformations."""
        transformations = []
        
        # Geometric transformations
        for op in ['rotate_90', 'rotate_180', 'rotate_270', 'flip_h', 'flip_v', 'transpose']:
            transformations.append(GeometricTransformation(op))
        
        # Scaling
        for factor in [2, 3]:
            transformations.append(GeometricTransformation('scale', {'factor': factor}))
        
        # Color transformations
        transformations.append(ColorTransformation('invert'))
        
        # Crop transformations
        transformations.append(CropTransformation('crop_content'))
        
        # Pattern transformations
        transformations.append(PatternTransformation('tile', {'rows': 2, 'cols': 2}))
        transformations.append(PatternTransformation('mirror', {'axis': 'h'}))
        transformations.append(PatternTransformation('mirror', {'axis': 'v'}))
        
        return transformations
    
    def discover_transformation(self, input_grid: np.ndarray, 
                                output_grid: np.ndarray) -> List[Tuple[Transformation, float]]:
        """
        Discover potential transformations from input to output.
        Returns list of (transformation, similarity_score) tuples.
        """
        candidates = []
        
        for transform in self.transformations:
            if transform.is_applicable(input_grid):
                try:
                    result = transform.apply(input_grid)
                    similarity = self._compute_similarity(result, output_grid)
                    if similarity > 0.1:
                        candidates.append((transform, similarity))
                except Exception:
                    continue
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def _compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Compute similarity between two grids."""
        # Handle different shapes
        if grid1.shape != grid2.shape:
            return 0.0
        
        # Exact match
        if np.array_equal(grid1, grid2):
            return 1.0
        
        # Cell-wise similarity
        matching = np.sum(grid1 == grid2)
        total = grid1.size
        
        return matching / total
    
    def compose_transformations(self, transforms: List[Transformation]) -> 'CompositeTransformation':
        """Compose multiple transformations into one."""
        return CompositeTransformation(transforms)


class CompositeTransformation(Transformation):
    """Composition of multiple transformations."""
    
    def __init__(self, transforms: List[Transformation]):
        self.transforms = transforms
    
    @property
    def name(self) -> str:
        return "composite_" + "_".join(t.name for t in self.transforms)
    
    def is_applicable(self, grid: np.ndarray) -> bool:
        return all(t.is_applicable(grid) for t in self.transforms)
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        for transform in self.transforms:
            result = transform.apply(result)
        return result


# =============================================================================
# SECTION 4: REASONING ENGINE
# =============================================================================

class ReasoningEngine:
    """
    Core reasoning engine for solving ARC tasks.
    Implements multiple reasoning strategies.
    """
    
    def __init__(self):
        self.pattern_recognizer = PatternRecognizer()
        self.transformation_engine = TransformationEngine()
        self.strategies = self._build_strategies()
    
    def _build_strategies(self) -> Dict[str, callable]:
        """Build reasoning strategies."""
        return {
            'pattern_matching': self._pattern_matching_strategy,
            'transformation_chain': self._transformation_chain_strategy,
            'object_manipulation': self._object_manipulation_strategy,
            'spatial_reasoning': self._spatial_reasoning_strategy,
            'analogy': self._analogy_strategy,
            'hypothesis_testing': self._hypothesis_testing_strategy
        }
    
    def solve(self, task: ARCTask) -> Tuple[List[np.ndarray], ReasoningTrace]:
        """
        Attempt to solve an ARC task using all available strategies.
        """
        trace = ReasoningTrace(task_id=task.task_id)
        solutions = []
        
        trace.add_step("Starting reasoning process")
        
        # Analyze training examples
        train_patterns = self._analyze_examples(task.train_examples, trace)
        trace.add_step("Analyzed training examples")
        
        # Try each strategy
        for strategy_name, strategy in self.strategies.items():
            trace.add_step(f"Trying strategy: {strategy_name}")
            
            try:
                predictions = strategy(task, train_patterns, trace)
                if predictions:
                    solutions.extend(predictions)
                    trace.add_step(f"Strategy {strategy_name} produced {len(predictions)} predictions")
            except Exception as e:
                trace.add_step(f"Strategy {strategy_name} failed: {str(e)}")
        
        trace.add_step(f"Reasoning complete. Generated {len(solutions)} potential solutions")
        
        return solutions, trace
    
    def _analyze_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                          trace: ReasoningTrace) -> Dict[str, Any]:
        """Analyze all training examples to extract patterns."""
        analysis = {
            'input_patterns': [],
            'output_patterns': [],
            'input_output_relations': [],
            'common_patterns': {},
            'size_relations': []
        }
        
        for i, (inp, out) in enumerate(examples):
            inp_patterns = self.pattern_recognizer.analyze(inp)
            out_patterns = self.pattern_recognizer.analyze(out) if out is not None else {}
            
            analysis['input_patterns'].append(inp_patterns)
            analysis['output_patterns'].append(out_patterns)
            
            # Analyze input-output relation
            relation = self._analyze_io_relation(inp, out)
            analysis['input_output_relations'].append(relation)
            
            # Track size changes
            analysis['size_relations'].append({
                'input_shape': inp.shape,
                'output_shape': out.shape if out is not None else None,
                'size_change': self._compute_size_change(inp, out)
            })
        
        # Find common patterns across examples
        analysis['common_patterns'] = self._find_common_patterns(analysis)
        
        return analysis
    
    def _analyze_io_relation(self, inp: np.ndarray, out: np.ndarray) -> Dict[str, Any]:
        """Analyze relation between input and output grids."""
        relation = {
            'same_shape': inp.shape == out.shape,
            'same_colors': set(np.unique(inp)) == set(np.unique(out)),
            'color_count_change': {},
            'structural_similarity': 0.0
        }
        
        # Color count changes
        inp_colors = Counter(inp.flatten())
        out_colors = Counter(out.flatten())
        
        all_colors = set(inp_colors.keys()) | set(out_colors.keys())
        for color in all_colors:
            relation['color_count_change'][int(color)] = (
                out_colors.get(color, 0) - inp_colors.get(color, 0)
            )
        
        # Structural similarity (if same shape)
        if inp.shape == out.shape:
            relation['structural_similarity'] = np.mean(inp == out)
        
        return relation
    
    def _compute_size_change(self, inp: np.ndarray, out: np.ndarray) -> Dict[str, Any]:
        """Compute size changes between input and output."""
        if out is None:
            return {'type': 'unknown'}
        
        inp_size = inp.shape
        out_size = out.shape
        
        row_ratio = out_size[0] / inp_size[0] if inp_size[0] > 0 else 0
        col_ratio = out_size[1] / inp_size[1] if inp_size[1] > 0 else 0
        
        return {
            'type': 'same' if inp_size == out_size else 'different',
            'row_ratio': row_ratio,
            'col_ratio': col_ratio,
            'area_ratio': (out_size[0] * out_size[1]) / (inp_size[0] * inp_size[1])
        }
    
    def _find_common_patterns(self, analysis: Dict) -> Dict[str, Any]:
        """Find patterns common across all examples."""
        common = {}
        
        # Check for consistent size changes
        size_changes = [s['size_change']['type'] for s in analysis['size_relations']]
        if len(set(size_changes)) == 1:
            common['consistent_size_change'] = size_changes[0]
        
        # Check for consistent transformations
        all_relations = analysis['input_output_relations']
        
        # Consistent shape relation
        if all(r['same_shape'] for r in all_relations):
            common['consistent_shape'] = 'same'
        elif not any(r['same_shape'] for r in all_relations):
            common['consistent_shape'] = 'different'
        
        return common
    
    def _pattern_matching_strategy(self, task: ARCTask, patterns: Dict, 
                                   trace: ReasoningTrace) -> List[np.ndarray]:
        """Pattern matching strategy: find and apply consistent patterns."""
        predictions = []
        
        # Check for consistent transformations
        common = patterns.get('common_patterns', {})
        
        # Get candidate transformations
        for inp, _ in task.test_examples:
            candidates = []
            
            for train_inp, train_out in task.train_examples:
                discovered = self.transformation_engine.discover_transformation(
                    train_inp, train_out
                )
                candidates.extend(discovered)
            
            # Find transformation that works for all training examples
            best_transforms = self._find_consistent_transforms(
                task.train_examples, candidates
            )
            
            for transform, score in best_transforms[:3]:
                pred = transform.apply(inp)
                predictions.append(pred)
                trace.hypothesis_scores[transform.name] = score
        
        return predictions
    
    def _find_consistent_transforms(self, train_examples: List[Tuple[np.ndarray, np.ndarray]],
                                    candidates: List[Tuple[Transformation, float]]) -> List[Tuple[Transformation, float]]:
        """Find transforms that work consistently across all examples."""
        consistent = []
        
        # Group by transformation
        transform_scores = defaultdict(list)
        for transform, score in candidates:
            transform_scores[transform.name].append((transform, score))
        
        # Score each unique transform
        for name, transforms in transform_scores.items():
            transform = transforms[0][0]
            total_score = 0
            
            for inp, out in train_examples:
                if transform.is_applicable(inp):
                    result = transform.apply(inp)
                    similarity = self.transformation_engine._compute_similarity(result, out)
                    total_score += similarity
            
            avg_score = total_score / len(train_examples)
            if avg_score > 0.5:
                consistent.append((transform, avg_score))
        
        consistent.sort(key=lambda x: x[1], reverse=True)
        return consistent
    
    def _transformation_chain_strategy(self, task: ARCTask, patterns: Dict,
                                       trace: ReasoningTrace) -> List[np.ndarray]:
        """Chain multiple transformations to solve the task."""
        predictions = []
        
        # Try to find transformation chains
        for inp, _ in task.test_examples:
            # Build potential chains
            chains = self._build_transformation_chains(inp, task.train_examples)
            
            for chain, score in chains[:2]:
                pred = chain.apply(inp)
                predictions.append(pred)
                trace.applied_transformations.append(chain.name)
        
        return predictions
    
    def _build_transformation_chains(self, test_input: np.ndarray,
                                     train_examples: List[Tuple[np.ndarray, np.ndarray]],
                                     max_depth: int = 3) -> List[Tuple[CompositeTransformation, float]]:
        """Build chains of transformations."""
        chains = []
        
        # Simple approach: try pairs of transformations
        transforms = self.transformation_engine.transformations
        
        for t1 in transforms:
            for t2 in transforms:
                if t1.is_applicable(test_input):
                    try:
                        intermediate = t1.apply(test_input)
                        if t2.is_applicable(intermediate):
                            chain = CompositeTransformation([t1, t2])
                            
                            # Score chain on training examples
                            score = 0
                            for inp, out in train_examples:
                                if chain.is_applicable(inp):
                                    result = chain.apply(inp)
                                    score += self.transformation_engine._compute_similarity(result, out)
                            
                            if score > 0:
                                chains.append((chain, score / len(train_examples)))
                    except Exception:
                        continue
        
        chains.sort(key=lambda x: x[1], reverse=True)
        return chains[:10]
    
    def _object_manipulation_strategy(self, task: ARCTask, patterns: Dict,
                                      trace: ReasoningTrace) -> List[np.ndarray]:
        """Strategy based on object identification and manipulation."""
        predictions = []
        
        for inp, _ in task.test_examples:
            # Identify objects
            objects = self._identify_objects(inp)
            trace.add_step(f"Identified {len(objects)} objects in test input")
            
            # Analyze how objects were manipulated in training
            object_rules = self._extract_object_rules(task.train_examples)
            
            # Apply rules to test objects
            pred = self._apply_object_rules(inp, objects, object_rules)
            if pred is not None:
                predictions.append(pred)
        
        return predictions
    
    def _identify_objects(self, grid: np.ndarray) -> List[Dict]:
        """Identify distinct objects in a grid."""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0 and not visited[i, j]:
                    obj = self._extract_object(grid, i, j, visited)
                    objects.append(obj)
        
        return objects
    
    def _extract_object(self, grid: np.ndarray, start_i: int, start_j: int,
                        visited: np.ndarray) -> Dict:
        """Extract a single connected object."""
        color = grid[start_i, start_j]
        cells = []
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            if (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and 
                not visited[i, j] and grid[i, j] == color):
                visited[i, j] = True
                cells.append((i, j))
                stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
        
        cells = np.array(cells)
        return {
            'color': int(color),
            'cells': cells,
            'centroid': cells.mean(axis=0),
            'size': len(cells),
            'bbox': (cells[:, 0].min(), cells[:, 1].min(),
                    cells[:, 0].max(), cells[:, 1].max())
        }
    
    def _extract_object_rules(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
        """Extract rules about object manipulation from examples."""
        rules = []
        
        for inp, out in examples:
            inp_objects = self._identify_objects(inp)
            out_objects = self._identify_objects(out) if out is not None else []
            
            rule = {
                'input_count': len(inp_objects),
                'output_count': len(out_objects),
                'size_changes': [],
                'color_changes': [],
                'position_changes': []
            }
            
            # Match objects and track changes
            for inp_obj in inp_objects:
                for out_obj in out_objects:
                    if inp_obj['color'] == out_obj['color']:
                        rule['size_changes'].append({
                            'color': inp_obj['color'],
                            'before': inp_obj['size'],
                            'after': out_obj['size']
                        })
                        rule['position_changes'].append({
                            'color': inp_obj['color'],
                            'centroid_shift': out_obj['centroid'] - inp_obj['centroid']
                        })
            
            rules.append(rule)
        
        return rules
    
    def _apply_object_rules(self, grid: np.ndarray, objects: List[Dict],
                           rules: List[Dict]) -> Optional[np.ndarray]:
        """Apply object manipulation rules to produce prediction."""
        if not rules:
            return None
        
        # Simple rule: check if all examples have same count change pattern
        count_changes = [r['output_count'] - r['input_count'] for r in rules]
        
        if len(set(count_changes)) == 1:
            change = count_changes[0]
            
            if change == 0:
                # Same number of objects - might be movement or color change
                return self._apply_same_count_rule(grid, objects, rules)
            elif change > 0:
                # Objects added
                return self._apply_addition_rule(grid, objects, rules, change)
            else:
                # Objects removed
                return self._apply_removal_rule(grid, objects, rules, -change)
        
        return None
    
    def _apply_same_count_rule(self, grid: np.ndarray, objects: List[Dict],
                               rules: List[Dict]) -> np.ndarray:
        """Apply rule when object count stays the same."""
        result = grid.copy()
        
        # Check for consistent position changes
        if rules and rules[0]['position_changes']:
            for obj in objects:
                for rule in rules:
                    for pos_change in rule['position_changes']:
                        if pos_change['color'] == obj['color']:
                            shift = pos_change['centroid_shift']
                            # Apply shift
                            for cell in obj['cells']:
                                result[cell[0], cell[1]] = 0
                            
                            new_cells = obj['cells'] + shift.astype(int)
                            for new_cell in new_cells:
                                if (0 <= new_cell[0] < result.shape[0] and
                                    0 <= new_cell[1] < result.shape[1]):
                                    result[new_cell[0], new_cell[1]] = obj['color']
        
        return result
    
    def _apply_addition_rule(self, grid: np.ndarray, objects: List[Dict],
                             rules: List[Dict], count: int) -> np.ndarray:
        """Apply rule when objects are added."""
        result = grid.copy()
        
        # Simple approach: duplicate existing objects
        if objects and count > 0:
            for obj in objects[:count]:
                # Mirror the object
                for cell in obj['cells']:
                    mirror_row = grid.shape[0] - 1 - cell[0]
                    if 0 <= mirror_row < result.shape[0]:
                        result[mirror_row, cell[1]] = obj['color']
        
        return result
    
    def _apply_removal_rule(self, grid: np.ndarray, objects: List[Dict],
                            rules: List[Dict], count: int) -> np.ndarray:
        """Apply rule when objects are removed."""
        result = grid.copy()
        
        # Remove smallest objects
        sorted_objects = sorted(objects, key=lambda x: x['size'])
        for obj in sorted_objects[:count]:
            for cell in obj['cells']:
                result[cell[0], cell[1]] = 0
        
        return result
    
    def _spatial_reasoning_strategy(self, task: ARCTask, patterns: Dict,
                                    trace: ReasoningTrace) -> List[np.ndarray]:
        """Strategy based on spatial reasoning."""
        predictions = []
        
        # Analyze spatial patterns
        spatial_patterns = self._analyze_spatial_patterns(task.train_examples, trace)
        
        for inp, _ in task.test_examples:
            pred = self._apply_spatial_reasoning(inp, spatial_patterns)
            if pred is not None:
                predictions.append(pred)
        
        return predictions
    
    def _analyze_spatial_patterns(self, examples: List[Tuple[np.ndarray, np.ndarray]],
                                  trace: ReasoningTrace) -> Dict[str, Any]:
        """Analyze spatial patterns across examples."""
        patterns = {
            'symmetry_type': None,
            'fill_patterns': [],
            'boundary_rules': []
        }
        
        # Check for symmetry patterns
        symmetry_counts = defaultdict(int)
        
        for inp, out in examples:
            if out is not None:
                symmetry = self.pattern_recognizer._detect_symmetry(out)
                for sym_type, is_present in symmetry.items():
                    if is_present:
                        symmetry_counts[sym_type] += 1
        
        if symmetry_counts:
            patterns['symmetry_type'] = max(symmetry_counts, key=symmetry_counts.get)
            trace.add_step(f"Detected symmetry pattern: {patterns['symmetry_type']}")
        
        return patterns
    
    def _apply_spatial_reasoning(self, grid: np.ndarray, 
                                  spatial_patterns: Dict) -> Optional[np.ndarray]:
        """Apply spatial reasoning to generate prediction."""
        result = grid.copy()
        
        # Apply symmetry if detected
        if spatial_patterns['symmetry_type']:
            sym = spatial_patterns['symmetry_type']
            
            if sym == 'horizontal':
                result = np.concatenate([result, np.flip(result, axis=0)], axis=0)
            elif sym == 'vertical':
                result = np.concatenate([result, np.flip(result, axis=1)], axis=1)
            elif sym == 'diagonal':
                result = result + result.T
        
        return result
    
    def _analogy_strategy(self, task: ARCTask, patterns: Dict,
                          trace: ReasoningTrace) -> List[np.ndarray]:
        """Strategy based on analogical reasoning."""
        predictions = []
        
        # Build analogy from training examples
        analogy = self._build_analogy(task.train_examples, trace)
        
        for inp, _ in task.test_examples:
            pred = self._apply_analogy(inp, analogy)
            if pred is not None:
                predictions.append(pred)
        
        return predictions
    
    def _build_analogy(self, examples: List[Tuple[np.ndarray, np.ndarray]],
                       trace: ReasoningTrace) -> Dict[str, Any]:
        """Build an analogy from training examples."""
        analogy = {
            'input_features': [],
            'output_features': [],
            'mapping': {}
        }
        
        for inp, out in examples:
            # Extract features
            inp_features = self._extract_features(inp)
            out_features = self._extract_features(out) if out is not None else {}
            
            analogy['input_features'].append(inp_features)
            analogy['output_features'].append(out_features)
        
        # Find mapping between features
        analogy['mapping'] = self._find_feature_mapping(
            analogy['input_features'],
            analogy['output_features']
        )
        
        trace.add_step(f"Built analogy with {len(analogy['mapping'])} feature mappings")
        
        return analogy
    
    def _extract_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract high-level features from a grid."""
        return {
            'shape': grid.shape,
            'colors': list(np.unique(grid)),
            'num_objects': len(self._identify_objects(grid)),
            'symmetry': self.pattern_recognizer._detect_symmetry(grid),
            'density': np.mean(grid > 0),
            'aspect_ratio': grid.shape[1] / grid.shape[0] if grid.shape[0] > 0 else 0
        }
    
    def _find_feature_mapping(self, inp_features: List[Dict], 
                              out_features: List[Dict]) -> Dict[str, Any]:
        """Find mapping from input features to output features."""
        mapping = {}
        
        # Map shape changes
        shapes_in = [f['shape'] for f in inp_features]
        shapes_out = [f['shape'] for f in out_features]
        
        if len(set(shapes_in)) == 1 and len(set(shapes_out)) == 1:
            mapping['shape_transform'] = {
                'from': shapes_in[0],
                'to': shapes_out[0]
            }
        
        # Map color changes
        colors_in = [set(f['colors']) for f in inp_features]
        colors_out = [set(f['colors']) for f in out_features]
        
        added_colors = set.union(*colors_out) - set.union(*colors_in)
        removed_colors = set.union(*colors_in) - set.union(*colors_out)
        
        if added_colors:
            mapping['colors_added'] = list(added_colors)
        if removed_colors:
            mapping['colors_removed'] = list(removed_colors)
        
        return mapping
    
    def _apply_analogy(self, grid: np.ndarray, analogy: Dict) -> Optional[np.ndarray]:
        """Apply analogy to generate prediction."""
        result = grid.copy()
        mapping = analogy['mapping']
        
        # Apply shape transform
        if 'shape_transform' in mapping:
            from_shape = mapping['shape_transform']['from']
            to_shape = mapping['shape_transform']['to']
            
            # Simple scaling
            row_scale = to_shape[0] / from_shape[0] if from_shape[0] > 0 else 1
            col_scale = to_shape[1] / from_shape[1] if from_shape[1] > 0 else 1
            
            if row_scale == col_scale and row_scale > 1:
                result = np.repeat(np.repeat(result, int(row_scale), axis=0), 
                                  int(col_scale), axis=1)
        
        # Apply color changes
        if 'colors_added' in mapping:
            # This is simplified - real implementation would need position logic
            pass
        
        return result
    
    def _hypothesis_testing_strategy(self, task: ARCTask, patterns: Dict,
                                     trace: ReasoningTrace) -> List[np.ndarray]:
        """Strategy based on hypothesis generation and testing."""
        predictions = []
        
        # Generate hypotheses
        hypotheses = self._generate_hypotheses(task.train_examples, trace)
        
        # Test hypotheses on training data
        valid_hypotheses = self._test_hypotheses(hypotheses, task.train_examples)
        
        # Apply valid hypotheses to test inputs
        for inp, _ in task.test_examples:
            for hypothesis in valid_hypotheses:
                pred = hypothesis.apply(inp)
                predictions.append(pred)
                trace.hypothesis_scores[hypothesis.name] = hypothesis.score
        
        return predictions
    
    def _generate_hypotheses(self, examples: List[Tuple[np.ndarray, np.ndarray]],
                            trace: ReasoningTrace) -> List['Hypothesis']:
        """Generate hypotheses about the transformation."""
        hypotheses = []
        
        # Hypothesis 1: Single transformation
        for inp, out in examples:
            candidates = self.transformation_engine.discover_transformation(inp, out)
            for transform, score in candidates:
                hypotheses.append(Hypothesis(
                    name=f"single_{transform.name}",
                    transformation=transform,
                    score=score
                ))
        
        # Hypothesis 2: Composite transformations
        for inp, out in examples:
            chains = self._build_transformation_chains(inp, [(inp, out)], max_depth=2)
            for chain, score in chains:
                hypotheses.append(Hypothesis(
                    name=chain.name,
                    transformation=chain,
                    score=score
                ))
        
        trace.add_step(f"Generated {len(hypotheses)} hypotheses")
        
        return hypotheses
    
    def _test_hypotheses(self, hypotheses: List['Hypothesis'],
                        examples: List[Tuple[np.ndarray, np.ndarray]]) -> List['Hypothesis']:
        """Test hypotheses on training examples and return valid ones."""
        valid = []
        
        for hypothesis in hypotheses:
            total_score = 0
            
            for inp, out in examples:
                try:
                    if hypothesis.transformation.is_applicable(inp):
                        result = hypothesis.transformation.apply(inp)
                        score = self.transformation_engine._compute_similarity(result, out)
                        total_score += score
                except Exception:
                    pass
            
            avg_score = total_score / len(examples) if examples else 0
            hypothesis.score = avg_score
            
            if avg_score > 0.7:
                valid.append(hypothesis)
        
        valid.sort(key=lambda h: h.score, reverse=True)
        return valid[:5]


@dataclass
class Hypothesis:
    """Represents a hypothesis about the transformation."""
    name: str
    transformation: Transformation
    score: float = 0.0
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        return self.transformation.apply(grid)


# =============================================================================
# SECTION 5: EVALUATION FRAMEWORK
# =============================================================================

class ARCEvaluator:
    """
    Evaluation framework for ARC solutions.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_solution(self, predicted: np.ndarray, 
                          expected: np.ndarray) -> Dict[str, float]:
        """Evaluate a single prediction against expected output."""
        metrics = {
            'exact_match': float(np.array_equal(predicted, expected)),
            'shape_match': float(predicted.shape == expected.shape),
            'cell_accuracy': self._cell_accuracy(predicted, expected),
            'color_accuracy': self._color_accuracy(predicted, expected),
            'iou': self._iou(predicted, expected),
            'size_accuracy': self._size_accuracy(predicted, expected)
        }
        return metrics
    
    def _cell_accuracy(self, predicted: np.ndarray, expected: np.ndarray) -> float:
        """Calculate cell-wise accuracy."""
        if predicted.shape != expected.shape:
            # Resize to match
            min_rows = min(predicted.shape[0], expected.shape[0])
            min_cols = min(predicted.shape[1], expected.shape[1])
            predicted = predicted[:min_rows, :min_cols]
            expected = expected[:min_rows, :min_cols]
        
        return float(np.mean(predicted == expected))
    
    def _color_accuracy(self, predicted: np.ndarray, expected: np.ndarray) -> float:
        """Calculate color distribution accuracy."""
        pred_colors = Counter(predicted.flatten())
        exp_colors = Counter(expected.flatten())
        
        all_colors = set(pred_colors.keys()) | set(exp_colors.keys())
        
        if not all_colors:
            return 1.0
        
        total_diff = sum(abs(pred_colors.get(c, 0) - exp_colors.get(c, 0)) 
                        for c in all_colors)
        total_cells = max(predicted.size, expected.size)
        
        return 1.0 - (total_diff / (2 * total_cells)) if total_cells > 0 else 0.0
    
    def _iou(self, predicted: np.ndarray, expected: np.ndarray) -> float:
        """Calculate Intersection over Union for non-zero cells."""
        pred_mask = predicted > 0
        exp_mask = expected > 0
        
        intersection = np.logical_and(pred_mask, exp_mask).sum()
        union = np.logical_or(pred_mask, exp_mask).sum()
        
        return float(intersection / union) if union > 0 else 0.0
    
    def _size_accuracy(self, predicted: np.ndarray, expected: np.ndarray) -> float:
        """Calculate size accuracy."""
        pred_size = predicted.shape[0] * predicted.shape[1]
        exp_size = expected.shape[0] * expected.shape[1]
        
        if exp_size == 0:
            return float(pred_size == 0)
        
        return 1.0 - abs(pred_size - exp_size) / exp_size
    
    def evaluate_task(self, task: ARCTask, 
                      predictions: List[List[np.ndarray]]) -> Dict[str, Any]:
        """
        Evaluate predictions for a task.
        predictions: list of lists, one list per test example with up to 3 attempts
        """
        results = {
            'task_id': task.task_id,
            'num_test': task.num_test,
            'example_results': [],
            'solved': True,
            'best_attempt_scores': []
        }
        
        for i, (test_input, test_output) in enumerate(task.test_examples):
            if test_output is None:
                continue
            
            example_preds = predictions[i] if i < len(predictions) else []
            example_metrics = []
            
            for pred in example_preds[:3]:  # Max 3 attempts
                metrics = self.evaluate_solution(pred, test_output)
                example_metrics.append(metrics)
            
            results['example_results'].append(example_metrics)
            
            # Check if solved
            if example_metrics:
                best_score = max(m['exact_match'] for m in example_metrics)
                results['best_attempt_scores'].append(best_score)
                if best_score < 1.0:
                    results['solved'] = False
            else:
                results['solved'] = False
        
        return results
    
    def evaluate_dataset(self, tasks: List[ARCTask],
                        all_predictions: List[List[List[np.ndarray]]]) -> Dict[str, Any]:
        """Evaluate a full dataset."""
        total_tasks = len(tasks)
        solved_tasks = 0
        all_results = []
        
        for task, predictions in zip(tasks, all_predictions):
            result = self.evaluate_task(task, predictions)
            all_results.append(result)
            if result['solved']:
                solved_tasks += 1
        
        return {
            'total_tasks': total_tasks,
            'solved_tasks': solved_tasks,
            'accuracy': solved_tasks / total_tasks if total_tasks > 0 else 0,
            'detailed_results': all_results
        }


# =============================================================================
# SECTION 6: VISUALIZATION MODULE
# =============================================================================

class ARCVisualizer:
    """
    Visualization tools for ARC tasks and solutions.
    """
    
    # ARC color palette
    COLORS = [
        '#000000',  # 0: black
        '#0074D9',  # 1: blue
        '#FF4136',  # 2: red
        '#2ECC40',  # 3: green
        '#FFDC00',  # 4: yellow
        '#AAAAAA',  # 5: gray
        '#F012BE',  # 6: magenta
        '#FF851B',  # 7: orange
        '#7FDBFF',  # 8: cyan
        '#870C25',  # 9: brown
    ]
    
    def __init__(self):
        self.cmap = ListedColormap(self.COLORS)
    
    def visualize_grid(self, grid: np.ndarray, ax=None, title: str = None):
        """Visualize a single grid."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.imshow(grid, cmap=self.cmap, vmin=0, vmax=9)
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if title:
            ax.set_title(title)
        
        return ax
    
    def visualize_task(self, task: ARCTask, figsize: Tuple[int, int] = None):
        """Visualize a complete ARC task."""
        n_train = task.num_train
        n_test = task.num_test
        
        if figsize is None:
            figsize = (4 * (n_train + n_test), 8)
        
        fig, axes = plt.subplots(2, n_train + n_test, figsize=figsize)
        
        # Training examples
        for i, (inp, out) in enumerate(task.train_examples):
            self.visualize_grid(inp, axes[0, i], f'Train {i+1} Input')
            self.visualize_grid(out, axes[1, i], f'Train {i+1} Output')
        
        # Test examples
        for i, (inp, out) in enumerate(task.test_examples):
            self.visualize_grid(inp, axes[0, n_train + i], f'Test {i+1} Input')
            if out is not None:
                self.visualize_grid(out, axes[1, n_train + i], f'Test {i+1} Output')
            else:
                axes[1, n_train + i].text(0.5, 0.5, '?', ha='center', va='center',
                                          fontsize=30, transform=axes[1, n_train + i].transAxes)
                axes[1, n_train + i].set_title(f'Test {i+1} Output (Unknown)')
        
        plt.tight_layout()
        return fig
    
    def visualize_solution(self, task: ARCTask, predictions: List[np.ndarray],
                          figsize: Tuple[int, int] = None):
        """Visualize a solution with predictions."""
        n_test = task.num_test
        
        if figsize is None:
            figsize = (6 * n_test, 12)
        
        fig, axes = plt.subplots(3, n_test, figsize=figsize)
        if n_test == 1:
            axes = axes.reshape(3, 1)
        
        for i, (inp, out) in enumerate(task.test_examples):
            self.visualize_grid(inp, axes[0, i], f'Test {i+1} Input')
            
            if out is not None:
                self.visualize_grid(out, axes[1, i], f'Test {i+1} Expected')
            else:
                axes[1, i].text(0.5, 0.5, '?', ha='center', va='center',
                               fontsize=30, transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'Test {i+1} Expected')
            
            if i < len(predictions):
                self.visualize_grid(predictions[i], axes[2, i], f'Test {i+1} Predicted')
            else:
                axes[2, i].text(0.5, 0.5, 'No prediction', ha='center', va='center',
                               transform=axes[2, i].transAxes)
                axes[2, i].set_title(f'Test {i+1} Predicted')
        
        plt.tight_layout()
        return fig
    
    def visualize_reasoning_trace(self, trace: ReasoningTrace):
        """Visualize reasoning trace."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # Hypothesis scores
        if trace.hypothesis_scores:
            names = list(trace.hypothesis_scores.keys())
            scores = list(trace.hypothesis_scores.values())
            
            axes[0].barh(range(len(names)), scores)
            axes[0].set_yticks(range(len(names)))
            axes[0].set_yticklabels(names, fontsize=8)
            axes[0].set_xlabel('Score')
            axes[0].set_title('Hypothesis Scores')
            axes[0].set_xlim(0, 1)
        
        # Reasoning steps timeline
        steps = trace.reasoning_steps
        axes[1].barh(range(len(steps)), [1] * len(steps))
        axes[1].set_yticks(range(len(steps)))
        axes[1].set_yticklabels([s[:50] + '...' if len(s) > 50 else s 
                                for s in steps], fontsize=8)
        axes[1].set_xlabel('Step')
        axes[1].set_title('Reasoning Steps')
        
        plt.tight_layout()
        return fig


# =============================================================================
# SECTION 7: DATASET HANDLING
# =============================================================================

class ARCDataLoader:
    """
    Loader for ARC dataset.
    """
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.tasks = {}
    
    def load_task(self, task_file: str) -> ARCTask:
        """Load a single task from JSON file."""
        with open(task_file, 'r') as f:
            data = json.load(f)
        
        task_id = task_file.split('/')[-1].replace('.json', '')
        
        train_examples = []
        for example in data.get('train', []):
            inp = np.array(example['input'])
            out = np.array(example['output'])
            train_examples.append((inp, out))
        
        test_examples = []
        for example in data.get('test', []):
            inp = np.array(example['input'])
            out = np.array(example.get('output')) if 'output' in example else None
            test_examples.append((inp, out))
        
        return ARCTask(task_id=task_id, train_examples=train_examples, 
                       test_examples=test_examples)
    
    def load_dataset(self, path: str) -> List[ARCTask]:
        """Load all tasks from a directory."""
        import os
        import glob
        
        tasks = []
        task_files = glob.glob(os.path.join(path, '*.json'))
        
        for task_file in task_files:
            try:
                task = self.load_task(task_file)
                tasks.append(task)
            except Exception as e:
                print(f"Error loading {task_file}: {e}")
        
        return tasks
    
    def create_sample_task(self) -> ARCTask:
        """Create a sample task for testing."""
        # Simple pattern: rotate 90 degrees
        train_examples = [
            (np.array([[1, 0], [1, 0]]), np.array([[0, 0], [1, 1]])),
            (np.array([[2, 0, 0], [2, 0, 0]]), np.array([[0, 0], [0, 0], [2, 2]])),
            (np.array([[3, 0], [3, 0], [3, 0]]), np.array([[0, 0, 0], [3, 3, 3]]))
        ]
        
        test_examples = [
            (np.array([[4, 0], [4, 0]]), np.array([[0, 0], [4, 4]]))
        ]
        
        return ARCTask(task_id='sample_rotation', train_examples=train_examples,
                      test_examples=test_examples)
    
    def create_sample_task_2(self) -> ARCTask:
        """Create another sample task for testing."""
        # Pattern: fill enclosed region
        train_examples = [
            (np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), 
             np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])),
            (np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]]),
             np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 1, 1, 1]]))
        ]
        
        test_examples = [
            (np.array([[3, 3, 3, 3], [3, 0, 0, 3], [3, 0, 0, 3], [3, 3, 3, 3]]),
             np.array([[3, 3, 3, 3], [3, 4, 4, 3], [3, 4, 4, 3], [3, 3, 3, 3]]))
        ]
        
        return ARCTask(task_id='sample_fill', train_examples=train_examples,
                      test_examples=test_examples)


# =============================================================================
# SECTION 8: ADVANCED REASONING MODULES
# =============================================================================

class MetaLearner:
    """
    Meta-learning module for learning from multiple tasks.
    """
    
    def __init__(self):
        self.learned_patterns = {}
        self.task_history = []
    
    def learn_from_task(self, task: ARCTask, solution_method: str, 
                       success: bool):
        """Learn from solving a task."""
        self.task_history.append({
            'task_id': task.task_id,
            'method': solution_method,
            'success': success,
            'patterns': self._extract_task_patterns(task)
        })
        
        # Update learned patterns
        if success:
            self._update_learned_patterns(task)
    
    def _extract_task_patterns(self, task: ARCTask) -> Dict[str, Any]:
        """Extract patterns from a task."""
        recognizer = PatternRecognizer()
        
        patterns = {
            'input_patterns': [recognizer.analyze(ex[0]) for ex in task.train_examples],
            'output_patterns': [recognizer.analyze(ex[1]) for ex in task.train_examples if ex[1] is not None]
        }
        
        return patterns
    
    def _update_learned_patterns(self, task: ARCTask):
        """Update the learned pattern database."""
        # Simple pattern aggregation
        for inp, out in task.train_examples:
            if out is not None:
                key = self._get_pattern_key(inp, out)
                if key not in self.learned_patterns:
                    self.learned_patterns[key] = 0
                self.learned_patterns[key] += 1
    
    def _get_pattern_key(self, inp: np.ndarray, out: np.ndarray) -> str:
        """Generate a key for a pattern."""
        return f"{inp.shape}_{out.shape}_{tuple(np.unique(inp))}_{tuple(np.unique(out))}"
    
    def suggest_method(self, task: ARCTask) -> List[str]:
        """Suggest solution methods based on learned patterns."""
        suggestions = []
        
        # Check for similar patterns
        for inp, out in task.train_examples:
            if out is not None:
                key = self._get_pattern_key(inp, out)
                if key in self.learned_patterns:
                    suggestions.append('pattern_matching')
        
        # Check task history for similar tasks
        for hist in self.task_history:
            if hist['success']:
                similarity = self._compute_task_similarity(task, hist['task_id'])
                if similarity > 0.5:
                    suggestions.append(hist['method'])
        
        return list(set(suggestions))
    
    def _compute_task_similarity(self, task1: ARCTask, task2_id: str) -> float:
        """Compute similarity between tasks."""
        # Simplified similarity computation
        hist_task = next((h for h in self.task_history if h['task_id'] == task2_id), None)
        if hist_task is None:
            return 0.0
        
        # Compare basic features
        # This is a placeholder for a more sophisticated comparison
        return 0.5


class ProgramSynthesizer:
    """
    Program synthesis module for generating transformation programs.
    """
    
    def __init__(self):
        self.primitives = self._build_primitives()
        self.max_program_length = 5
    
    def _build_primitives(self) -> Dict[str, callable]:
        """Build library of primitive operations."""
        return {
            'rotate_90': lambda g: np.rot90(g, 1),
            'rotate_180': lambda g: np.rot90(g, 2),
            'rotate_270': lambda g: np.rot90(g, 3),
            'flip_h': lambda g: np.flip(g, axis=0),
            'flip_v': lambda g: np.flip(g, axis=1),
            'transpose': lambda g: g.T,
            'scale_2': lambda g: np.repeat(np.repeat(g, 2, axis=0), 2, axis=1),
            'scale_3': lambda g: np.repeat(np.repeat(g, 3, axis=0), 3, axis=1),
            'crop_content': self._crop_content,
            'fill_bg_1': lambda g: np.where(g == 0, 1, g),
            'invert_binary': lambda g: np.where(g > 0, 0, 1) if set(np.unique(g)) <= {0, 1} else g
        }
    
    def _crop_content(self, grid: np.ndarray) -> np.ndarray:
        """Crop to content."""
        rows = np.any(grid != 0, axis=1)
        cols = np.any(grid != 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return grid
        
        r_idx = np.where(rows)[0]
        c_idx = np.where(cols)[0]
        
        return grid[r_idx[0]:r_idx[-1]+1, c_idx[0]:c_idx[-1]+1]
    
    def synthesize(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
        """Synthesize programs that transform inputs to outputs."""
        programs = []
        
        # Try single primitives
        for name, prim in self.primitives.items():
            if self._test_primitive(name, prim, examples):
                programs.append({
                    'program': [name],
                    'score': 1.0
                })
        
        # Try pairs of primitives
        if not programs:
            for n1, p1 in self.primitives.items():
                for n2, p2 in self.primitives.items():
                    combined = lambda g, p1=p1, p2=p2: p2(p1(g))
                    if self._test_primitive(f"{n1}_{n2}", combined, examples):
                        programs.append({
                            'program': [n1, n2],
                            'score': 0.9
                        })
        
        return programs
    
    def _test_primitive(self, name: str, func: callable, 
                       examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Test if a primitive transforms all inputs to outputs."""
        for inp, out in examples:
            try:
                result = func(inp.copy())
                if not np.array_equal(result, out):
                    return False
            except Exception:
                return False
        return True
    
    def execute_program(self, program: List[str], grid: np.ndarray) -> np.ndarray:
        """Execute a synthesized program."""
        result = grid.copy()
        for step in program:
            if step in self.primitives:
                result = self.primitives[step](result)
        return result


# =============================================================================
# SECTION 9: MAIN SOLVER CLASS
# =============================================================================

class ARCSolver:
    """
    Main solver class that integrates all components.
    """
    
    def __init__(self):
        self.reasoning_engine = ReasoningEngine()
        self.evaluator = ARCEvaluator()
        self.visualizer = ARCVisualizer()
        self.meta_learner = MetaLearner()
        self.program_synthesizer = ProgramSynthesizer()
        self.data_loader = ARCDataLoader()
    
    def solve_task(self, task: ARCTask, visualize: bool = False) -> Dict[str, Any]:
        """
        Solve an ARC task.
        """
        print(f"\n{'='*60}")
        print(f"Solving task: {task.task_id}")
        print(f"Training examples: {task.num_train}")
        print(f"Test examples: {task.num_test}")
        print(f"{'='*60}")
        
        # Get suggestions from meta-learner
        suggested_methods = self.meta_learner.suggest_method(task)
        print(f"Suggested methods: {suggested_methods}")
        
        # Solve using reasoning engine
        solutions, trace = self.reasoning_engine.solve(task)
        
        # Try program synthesis
        synthesized_programs = self.program_synthesizer.synthesize(task.train_examples)
        for prog in synthesized_programs:
            for inp, _ in task.test_examples:
                pred = self.program_synthesizer.execute_program(prog['program'], inp)
                solutions.append(pred)
        
        # Organize predictions by test example
        predictions = self._organize_predictions(solutions, task.num_test)
        
        # Evaluate
        results = self.evaluator.evaluate_task(task, predictions)
        
        # Learn from this task
        self.meta_learner.learn_from_task(
            task, 
            trace.detected_patterns[0] if trace.detected_patterns else 'unknown',
            results['solved']
        )
        
        # Visualize
        if visualize:
            self.visualizer.visualize_task(task)
            if predictions:
                self.visualizer.visualize_solution(task, [p[0] for p in predictions])
            self.visualizer.visualize_reasoning_trace(trace)
            plt.show()
        
        return {
            'task_id': task.task_id,
            'predictions': predictions,
            'evaluation': results,
            'trace': trace
        }
    
    def _organize_predictions(self, solutions: List[np.ndarray], 
                              num_test: int) -> List[List[np.ndarray]]:
        """Organize solutions into predictions per test example."""
        predictions = [[] for _ in range(num_test)]
        
        for i in range(num_test):
            # Get up to 3 unique predictions for each test example
            unique_preds = []
            seen = set()
            
            for sol in solutions:
                key = sol.tobytes()
                if key not in seen and len(unique_preds) < 3:
                    unique_preds.append(sol)
                    seen.add(key)
            
            predictions[i] = unique_preds
        
        return predictions
    
    def solve_dataset(self, tasks: List[ARCTask]) -> Dict[str, Any]:
        """Solve a dataset of tasks."""
        all_predictions = []
        all_results = []
        
        for task in tasks:
            result = self.solve_task(task, visualize=False)
            all_predictions.append(result['predictions'])
            all_results.append(result['evaluation'])
        
        # Aggregate results
        solved = sum(1 for r in all_results if r['solved'])
        
        return {
            'total_tasks': len(tasks),
            'solved_tasks': solved,
            'accuracy': solved / len(tasks) if tasks else 0,
            'detailed_results': all_results
        }


# =============================================================================
# SECTION 10: EXPERIMENTAL FEATURES
# =============================================================================

class NeuralPatternEncoder:
    """
    Experimental: Neural network-based pattern encoding.
    """
    
    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        self.encoder = self._build_encoder()
    
    def _build_encoder(self):
        """Build the encoder architecture."""
        # This is a placeholder for neural network architecture
        # In practice, this would use PyTorch or TensorFlow
        return {
            'type': 'cnn_encoder',
            'layers': ['conv1', 'conv2', 'fc'],
            'hidden_dim': self.hidden_dim
        }
    
    def encode(self, grid: np.ndarray) -> np.ndarray:
        """Encode a grid into a latent representation."""
        # Placeholder: simple flattening
        normalized = grid.astype(float) / 9.0
        return normalized.flatten()
    
    def decode(self, latent: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Decode latent representation to grid."""
        # Placeholder: reshape
        size = target_shape[0] * target_shape[1]
        if len(latent) >= size:
            grid = latent[:size].reshape(target_shape)
            return (grid * 9).astype(int)
        return np.zeros(target_shape, dtype=int)


class AttentionModule:
    """
    Experimental: Attention mechanism for focusing on important regions.
    """
    
    def __init__(self):
        self.attention_weights = None
    
    def compute_attention(self, grid: np.ndarray) -> np.ndarray:
        """Compute attention weights for a grid."""
        # Simple attention: focus on non-zero regions
        attention = (grid > 0).astype(float)
        
        # Add edge attention
        edge_attention = np.zeros_like(grid, dtype=float)
        edge_attention[0, :] = 1
        edge_attention[-1, :] = 1
        edge_attention[:, 0] = 1
        edge_attention[:, -1] = 1
        
        attention = attention + 0.3 * edge_attention
        attention = attention / attention.max() if attention.max() > 0 else attention
        
        self.attention_weights = attention
        return attention
    
    def get_focused_regions(self, grid: np.ndarray, 
                           threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """Get bounding boxes of high-attention regions."""
        if self.attention_weights is None:
            self.compute_attention(grid)
        
        attention = self.attention_weights
        mask = attention > threshold
        
        regions = []
        visited = np.zeros_like(mask, dtype=bool)
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and not visited[i, j]:
                    # Find connected region
                    region = self._flood_fill_attention(mask, i, j, visited)
                    if region:
                        regions.append(region)
        
        return regions
    
    def _flood_fill_attention(self, mask: np.ndarray, start_i: int, start_j: int,
                              visited: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Flood fill to find connected high-attention region."""
        cells = []
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            if (0 <= i < mask.shape[0] and 0 <= j < mask.shape[1] and 
                mask[i, j] and not visited[i, j]):
                visited[i, j] = True
                cells.append((i, j))
                stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
        
        if not cells:
            return None
        
        rows = [c[0] for c in cells]
        cols = [c[1] for c in cells]
        
        return (min(rows), min(cols), max(rows), max(cols))


# =============================================================================
# SECTION 11: BENCHMARK AND TESTING
# =============================================================================

class ARCBenchmark:
    """
    Benchmark suite for testing ARC solver capabilities.
    """
    
    def __init__(self):
        self.solver = ARCSolver()
        self.test_tasks = self._create_benchmark_tasks()
    
    def _create_benchmark_tasks(self) -> List[ARCTask]:
        """Create a set of benchmark tasks."""
        tasks = []
        
        # Task 1: Rotation
        tasks.append(ARCTask(
            task_id='rotate_90',
            train_examples=[
                (np.array([[1, 0], [1, 0]]), np.array([[0, 0], [1, 1]])),
                (np.array([[2, 0, 0], [2, 0, 0]]), np.array([[0, 0], [0, 0], [2, 2]])),
            ],
            test_examples=[
                (np.array([[3, 0, 0], [3, 0, 0], [3, 0, 0]]), 
                 np.array([[0, 0, 0], [3, 3, 3]]))
            ]
        ))
        
        # Task 2: Fill enclosed
        tasks.append(ARCTask(
            task_id='fill_enclosed',
            train_examples=[
                (np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), 
                 np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])),
            ],
            test_examples=[
                (np.array([[3, 3, 3, 3], [3, 0, 0, 3], [3, 0, 0, 3], [3, 3, 3, 3]]), None)
            ]
        ))
        
        # Task 3: Mirror
        tasks.append(ARCTask(
            task_id='mirror_horizontal',
            train_examples=[
                (np.array([[1, 0, 0]]), np.array([[1, 0, 0], [0, 0, 1]])),
                (np.array([[2, 2, 0]]), np.array([[2, 2, 0], [0, 2, 2]])),
            ],
            test_examples=[
                (np.array([[3, 0, 3]]), None)
            ]
        ))
        
        # Task 4: Scale
        tasks.append(ARCTask(
            task_id='scale_2x',
            train_examples=[
                (np.array([[1]]), np.array([[1, 1], [1, 1]])),
                (np.array([[1, 2], [3, 4]]), np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])),
            ],
            test_examples=[
                (np.array([[2, 0], [0, 1]]), None)
            ]
        ))
        
        # Task 5: Color replacement
        tasks.append(ARCTask(
            task_id='color_replace',
            train_examples=[
                (np.array([[1, 1, 2], [2, 1, 2]]), np.array([[3, 3, 2], [2, 3, 2]])),
                (np.array([[1, 2, 1]]), np.array([[3, 2, 3]])),
            ],
            test_examples=[
                (np.array([[1, 1, 1, 2]]), None)
            ]
        ))
        
        return tasks
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the benchmark suite."""
        print("\n" + "="*70)
        print("RUNNING ARC BENCHMARK")
        print("="*70)
        
        results = {
            'task_results': [],
            'summary': {}
        }
        
        for task in self.test_tasks:
            print(f"\nBenchmarking: {task.task_id}")
            result = self.solver.solve_task(task, visualize=False)
            results['task_results'].append(result)
        
        # Summary
        solved = sum(1 for r in results['task_results'] if r['evaluation']['solved'])
        total = len(results['task_results'])
        
        results['summary'] = {
            'total_tasks': total,
            'solved': solved,
            'accuracy': solved / total if total > 0 else 0,
            'unsolved_tasks': [r['task_id'] for r in results['task_results'] if not r['evaluation']['solved']]
        }
        
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        print(f"Total tasks: {results['summary']['total_tasks']}")
        print(f"Solved: {results['summary']['solved']}")
        print(f"Accuracy: {results['summary']['accuracy']:.2%}")
        if results['summary']['unsolved_tasks']:
            print(f"Unsolved: {results['summary']['unsolved_tasks']}")
        
        return results
    
    def profile_solver(self) -> Dict[str, float]:
        """Profile the solver to identify bottlenecks."""
        import time
        
        profile = {}
        
        # Test each component
        task = self.test_tasks[0]
        
        # Profile pattern recognition
        start = time.time()
        recognizer = PatternRecognizer()
        for _ in range(100):
            recognizer.analyze(task.train_examples[0][0])
        profile['pattern_recognition_100'] = time.time() - start
        
        # Profile transformation discovery
        start = time.time()
        engine = TransformationEngine()
        for _ in range(100):
            engine.discover_transformation(
                task.train_examples[0][0],
                task.train_examples[0][1]
            )
        profile['transformation_discovery_100'] = time.time() - start
        
        # Profile full solve
        start = time.time()
        self.solver.solve_task(task, visualize=False)
        profile['full_solve'] = time.time() - start
        
        return profile


# =============================================================================
# SECTION 12: UTILITY FUNCTIONS
# =============================================================================

def print_grid(grid: np.ndarray, title: str = None):
    """Pretty print a grid."""
    if title:
        print(f"\n{title}")
    print("-" * (grid.shape[1] * 2 + 1))
    for row in grid:
        print("|" + " ".join(str(c) if c != 0 else "." for c in row) + "|")
    print("-" * (grid.shape[1] * 2 + 1))


def grid_to_string(grid: np.ndarray) -> str:
    """Convert grid to string representation."""
    lines = []
    for row in grid:
        lines.append("".join(str(c) if c != 0 else "." for c in row))
    return "\n".join(lines)


def compare_grids(grid1: np.ndarray, grid2: np.ndarray) -> Dict[str, Any]:
    """Compare two grids and return differences."""
    comparison = {
        'same_shape': grid1.shape == grid2.shape,
        'exact_match': np.array_equal(grid1, grid2),
        'cell_differences': [],
        'color_differences': {}
    }
    
    if grid1.shape == grid2.shape:
        diff_mask = grid1 != grid2
        diff_positions = np.argwhere(diff_mask)
        
        comparison['cell_differences'] = [
            (int(p[0]), int(p[1]), int(grid1[p[0], p[1]]), int(grid2[p[0], p[1]]))
            for p in diff_positions
        ]
        
        comparison['num_differences'] = len(comparison['cell_differences'])
        comparison['similarity'] = 1 - (comparison['num_differences'] / grid1.size)
    else:
        comparison['num_differences'] = -1
        comparison['similarity'] = 0
    
    return comparison


def create_task_from_examples(examples: List[Tuple[List[List[int]], List[List[int]]]],
                              task_id: str = "custom") -> ARCTask:
    """Create an ARCTask from example lists."""
    train_examples = [(np.array(ex[0]), np.array(ex[1])) for ex in examples[:-1]]
    test_examples = [(np.array(examples[-1][0]), 
                      np.array(examples[-1][1]) if examples[-1][1] else None)]
    
    return ARCTask(task_id=task_id, train_examples=train_examples, 
                   test_examples=test_examples)


# =============================================================================
# SECTION 13: MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function demonstrating the ARC solver.
    """
    print("="*80)
    print("FROM PATTERN RECOGNITION TO REASONING: ARC PRIZE 2026")
    print("="*80)
    print("\nInitializing ARC Solver System...\n")
    
    # Initialize solver
    solver = ARCSolver()
    
    # Load sample tasks
    loader = ARCDataLoader()
    tasks = [
        loader.create_sample_task(),
        loader.create_sample_task_2()
    ]
    
    print("Loaded sample tasks for demonstration.\n")
    
    # Solve tasks
    for task in tasks:
        result = solver.solve_task(task, visualize=False)
        
        print(f"\nTask: {task.task_id}")
        print(f"Solved: {result['evaluation']['solved']}")
        
        if result['predictions']:
            print("\nPredictions generated:")
            for i, preds in enumerate(result['predictions']):
                print(f"  Test {i+1}: {len(preds)} prediction(s)")
                if preds:
                    print_grid(preds[0], f"  Best prediction for Test {i+1}")
        
        print(f"\nReasoning steps: {len(result['trace'].reasoning_steps)}")
        print(f"Detected patterns: {result['trace'].detected_patterns}")
    
    # Run benchmark
    print("\n" + "="*80)
    print("Running Benchmark Suite...")
    print("="*80)
    
    benchmark = ARCBenchmark()
    benchmark_results = benchmark.run_benchmark()
    
    # Profile solver
    print("\n" + "="*80)
    print("Profiling Solver Performance...")
    print("="*80)
    
    profile = benchmark.profile_solver()
    for metric, value in profile.items():
        print(f"{metric}: {value:.4f} seconds")
    
    print("\n" + "="*80)
    print("ARC PRIZE 2026 DEMONSTRATION COMPLETE")
    print("="*80)
    
    return solver, benchmark_results


if __name__ == "__main__":
    # Run main demonstration
    solver, results = main()
    
    # Interactive exploration example
    print("\n" + "="*80)
    print("INTERACTIVE EXPLORATION")
    print("="*80)
    
    # Create a custom task
    custom_task = create_task_from_examples([
        ([[1, 0, 1], [0, 0, 0], [1, 0, 1]], 
         [[1, 0, 1], [0, 2, 0], [1, 0, 1]]),
        ([[3, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 3]],
         [[3, 0, 0, 3], [0, 4, 4, 0], [0, 4, 4, 0], [3, 0, 0, 3]]),
        ([[5, 0, 5]], None)  # Test example
    ], task_id="custom_pattern")
    
    print(f"\nCreated custom task: {custom_task.task_id}")
    print("Training examples:")
    for i, (inp, out) in enumerate(custom_task.train_examples):
        print_grid(inp, f"Train {i+1} Input")
        print_grid(out, f"Train {i+1} Output")
    
    # Solve custom task
    custom_result = solver.solve_task(custom_task, visualize=False)
    print(f"\nCustom task solved: {custom_result['evaluation']['solved']}")
    
    if custom_result['predictions'] and custom_result['predictions'][0]:
        print_grid(custom_result['predictions'][0][0], "Prediction")
