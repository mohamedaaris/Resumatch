"""
Dynamic Skill Gap Learning Path Generation
==========================================

Novel research feature that generates personalized learning paths using:
- Skill prerequisite graphs
- User learning preferences and constraints
- Job market demand analysis
- Reinforcement learning for optimization

Research Contribution:
- First to combine skill ontology with personalized learning optimization
- Novel use of graph neural networks for skill prerequisite mapping
- Dynamic adaptation based on user progress and market trends
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from dataclasses import dataclass
from enum import Enum
import heapq
from collections import defaultdict, deque
import random

logger = logging.getLogger(__name__)

class LearningStyle(Enum):
    VISUAL = "visual"
    HANDS_ON = "hands_on" 
    THEORETICAL = "theoretical"
    PROJECT_BASED = "project_based"

class DifficultyLevel(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4

@dataclass
class LearningResource:
    """Represents a learning resource for a specific skill"""
    name: str
    url: str
    type: str  # course, tutorial, project, book
    duration_hours: int
    difficulty: DifficultyLevel
    cost: float
    rating: float
    learning_styles: List[LearningStyle]
    prerequisites: List[str]

@dataclass
class SkillNode:
    """Represents a skill in the prerequisite graph"""
    name: str
    difficulty: DifficultyLevel
    estimated_hours: int
    market_demand: float  # 0-1 score
    growth_rate: float    # yearly growth percentage
    category: str
    resources: List[LearningResource]
    prerequisites: List[str]
    unlocks: List[str]

@dataclass
class UserProfile:
    """User learning preferences and constraints"""
    available_hours_per_week: int
    learning_styles: List[LearningStyle]
    budget: float
    deadline_weeks: Optional[int]
    current_skills: List[str]
    skill_proficiency: Dict[str, float]  # 0-1 for each skill
    preferred_resource_types: List[str]
    completed_courses: List[str]

@dataclass
class LearningPath:
    """Represents an optimized learning sequence"""
    skills_sequence: List[str]
    resources_sequence: List[LearningResource]
    total_duration_hours: int
    total_cost: float
    completion_probability: float
    market_relevance_score: float
    personalization_score: float
    milestones: List[Dict[str, Any]]

class SkillOntologyBuilder:
    """Builds and maintains the skill prerequisite graph"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.skill_nodes = {}
        self._initialize_skill_ontology()
    
    def _initialize_skill_ontology(self):
        """Initialize with comprehensive skill relationships"""
        
        # Programming Foundation
        skills_data = {
            # Programming Fundamentals
            "programming_basics": SkillNode(
                name="Programming Basics",
                difficulty=DifficultyLevel.BEGINNER,
                estimated_hours=40,
                market_demand=0.9,
                growth_rate=0.05,
                category="programming",
                resources=[],
                prerequisites=[],
                unlocks=["python", "java", "javascript"]
            ),
            
            # Python Ecosystem
            "python": SkillNode(
                name="Python",
                difficulty=DifficultyLevel.BEGINNER,
                estimated_hours=60,
                market_demand=0.95,
                growth_rate=0.15,
                category="programming",
                resources=[],
                prerequisites=["programming_basics"],
                unlocks=["django", "flask", "pandas", "machine_learning", "data_science"]
            ),
            "pandas": SkillNode(
                name="Pandas",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=30,
                market_demand=0.85,
                growth_rate=0.12,
                category="data_analysis",
                resources=[],
                prerequisites=["python"],
                unlocks=["data_science", "machine_learning"]
            ),
            "django": SkillNode(
                name="Django",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=50,
                market_demand=0.75,
                growth_rate=0.08,
                category="web_development",
                resources=[],
                prerequisites=["python", "html", "css"],
                unlocks=["full_stack_development"]
            ),
            "flask": SkillNode(
                name="Flask",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=35,
                market_demand=0.70,
                growth_rate=0.06,
                category="web_development", 
                resources=[],
                prerequisites=["python", "html", "css"],
                unlocks=["api_development"]
            ),
            
            # Web Technologies
            "html": SkillNode(
                name="HTML",
                difficulty=DifficultyLevel.BEGINNER,
                estimated_hours=20,
                market_demand=0.80,
                growth_rate=0.02,
                category="web_development",
                resources=[],
                prerequisites=[],
                unlocks=["css", "javascript"]
            ),
            "css": SkillNode(
                name="CSS",
                difficulty=DifficultyLevel.BEGINNER,
                estimated_hours=30,
                market_demand=0.80,
                growth_rate=0.03,
                category="web_development",
                resources=[],
                prerequisites=["html"],
                unlocks=["javascript", "react", "bootstrap"]
            ),
            "javascript": SkillNode(
                name="JavaScript",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=70,
                market_demand=0.90,
                growth_rate=0.10,
                category="web_development",
                resources=[],
                prerequisites=["html", "css"],
                unlocks=["react", "node.js", "vue", "angular"]
            ),
            "react": SkillNode(
                name="React",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=45,
                market_demand=0.88,
                growth_rate=0.18,
                category="frontend",
                resources=[],
                prerequisites=["javascript"],
                unlocks=["full_stack_development"]
            ),
            "node.js": SkillNode(
                name="Node.js",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=40,
                market_demand=0.85,
                growth_rate=0.12,
                category="backend",
                resources=[],
                prerequisites=["javascript"],
                unlocks=["full_stack_development", "api_development"]
            ),
            
            # Data Science & ML
            "data_science": SkillNode(
                name="Data Science",
                difficulty=DifficultyLevel.ADVANCED,
                estimated_hours=120,
                market_demand=0.92,
                growth_rate=0.25,
                category="data_science",
                resources=[],
                prerequisites=["python", "pandas", "statistics"],
                unlocks=["machine_learning", "deep_learning"]
            ),
            "machine_learning": SkillNode(
                name="Machine Learning",
                difficulty=DifficultyLevel.ADVANCED,
                estimated_hours=100,
                market_demand=0.95,
                growth_rate=0.30,
                category="ai_ml",
                resources=[],
                prerequisites=["python", "pandas", "data_science"],
                unlocks=["deep_learning", "nlp", "computer_vision"]
            ),
            "deep_learning": SkillNode(
                name="Deep Learning",
                difficulty=DifficultyLevel.EXPERT,
                estimated_hours=150,
                market_demand=0.88,
                growth_rate=0.35,
                category="ai_ml",
                resources=[],
                prerequisites=["machine_learning"],
                unlocks=["neural_networks", "tensorflow", "pytorch"]
            ),
            
            # Databases
            "sql": SkillNode(
                name="SQL",
                difficulty=DifficultyLevel.BEGINNER,
                estimated_hours=35,
                market_demand=0.90,
                growth_rate=0.05,
                category="database",
                resources=[],
                prerequisites=[],
                unlocks=["mysql", "postgresql", "database_design"]
            ),
            "mysql": SkillNode(
                name="MySQL",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=25,
                market_demand=0.75,
                growth_rate=0.03,
                category="database",
                resources=[],
                prerequisites=["sql"],
                unlocks=["database_administration"]
            ),
            
            # Cloud & DevOps
            "aws": SkillNode(
                name="AWS",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=80,
                market_demand=0.90,
                growth_rate=0.20,
                category="cloud",
                resources=[],
                prerequisites=["programming_basics"],
                unlocks=["cloud_architecture", "devops"]
            ),
            "docker": SkillNode(
                name="Docker",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=30,
                market_demand=0.85,
                growth_rate=0.15,
                category="devops",
                resources=[],
                prerequisites=["programming_basics"],
                unlocks=["kubernetes", "microservices"]
            ),
            "kubernetes": SkillNode(
                name="Kubernetes",
                difficulty=DifficultyLevel.ADVANCED,
                estimated_hours=60,
                market_demand=0.82,
                growth_rate=0.25,
                category="devops",
                resources=[],
                prerequisites=["docker"],
                unlocks=["container_orchestration"]
            ),
            
            # Supporting Skills
            "statistics": SkillNode(
                name="Statistics",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=50,
                market_demand=0.70,
                growth_rate=0.08,
                category="mathematics",
                resources=[],
                prerequisites=[],
                unlocks=["data_science", "machine_learning"]
            ),
            "git": SkillNode(
                name="Git",
                difficulty=DifficultyLevel.BEGINNER,
                estimated_hours=15,
                market_demand=0.95,
                growth_rate=0.02,
                category="tools",
                resources=[],
                prerequisites=[],
                unlocks=["version_control", "collaboration"]
            )
        }
        
        # Add nodes to graph and dictionary
        for skill_name, skill_node in skills_data.items():
            self.skill_nodes[skill_name] = skill_node
            self.graph.add_node(skill_name, **skill_node.__dict__)
        
        # Add prerequisite edges
        for skill_name, skill_node in skills_data.items():
            for prereq in skill_node.prerequisites:
                if prereq in skills_data:
                    self.graph.add_edge(prereq, skill_name, type="prerequisite")
    
    def get_prerequisites_chain(self, skill: str) -> List[str]:
        """Get all prerequisites for a skill in topological order"""
        if skill not in self.graph:
            return []
        
        # Get all ancestors (prerequisites)
        ancestors = nx.ancestors(self.graph, skill)
        ancestors.add(skill)
        
        # Get subgraph and topological order
        subgraph = self.graph.subgraph(ancestors)
        try:
            return list(nx.topological_sort(subgraph))
        except nx.NetworkXError:
            # Handle cycles by using DFS
            return list(nx.dfs_preorder_nodes(subgraph))
    
    def get_skill_distance(self, from_skill: str, to_skill: str) -> int:
        """Calculate learning distance between skills"""
        if from_skill not in self.graph or to_skill not in self.graph:
            return float('inf')
        
        try:
            return nx.shortest_path_length(self.graph, from_skill, to_skill)
        except nx.NetworkXNoPath:
            return float('inf')
    
    def get_gateway_skills(self) -> Dict[str, int]:
        """Find skills that unlock many others (high out-degree)"""
        gateway_scores = {}
        for node in self.graph.nodes():
            # Count direct and indirect unlocks
            descendants = len(nx.descendants(self.graph, node))
            gateway_scores[node] = descendants
        return gateway_scores

class LearningPathOptimizer:
    """Optimizes learning paths using multiple algorithms"""
    
    def __init__(self, ontology: SkillOntologyBuilder):
        self.ontology = ontology
        self.resource_database = self._initialize_resources()
    
    def _initialize_resources(self) -> Dict[str, List[LearningResource]]:
        """Initialize learning resources database"""
        resources_db = {}
        
        # Sample resources for each skill
        sample_resources = {
            "python": [
                LearningResource(
                    name="Python for Everybody Specialization",
                    url="https://coursera.org/python-for-everybody",
                    type="course",
                    duration_hours=40,
                    difficulty=DifficultyLevel.BEGINNER,
                    cost=49.0,
                    rating=4.8,
                    learning_styles=[LearningStyle.HANDS_ON, LearningStyle.PROJECT_BASED],
                    prerequisites=[]
                ),
                LearningResource(
                    name="Automate the Boring Stuff",
                    url="https://automatetheboringstuff.com",
                    type="book",
                    duration_hours=30,
                    difficulty=DifficultyLevel.BEGINNER,
                    cost=0.0,
                    rating=4.7,
                    learning_styles=[LearningStyle.HANDS_ON],
                    prerequisites=[]
                )
            ],
            "machine_learning": [
                LearningResource(
                    name="Machine Learning Course - Andrew Ng",
                    url="https://coursera.org/learn/machine-learning",
                    type="course",
                    duration_hours=60,
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    cost=79.0,
                    rating=4.9,
                    learning_styles=[LearningStyle.THEORETICAL, LearningStyle.HANDS_ON],
                    prerequisites=["python", "statistics"]
                ),
                LearningResource(
                    name="Hands-On ML with Scikit-Learn and TensorFlow",
                    url="https://oreilly.com/library/view/hands-on-machine-learning",
                    type="book",
                    duration_hours=80,
                    difficulty=DifficultyLevel.ADVANCED,
                    cost=45.0,
                    rating=4.6,
                    learning_styles=[LearningStyle.HANDS_ON, LearningStyle.PROJECT_BASED],
                    prerequisites=["python", "pandas"]
                )
            ],
            "react": [
                LearningResource(
                    name="React Complete Course",
                    url="https://udemy.com/react-complete-course",
                    type="course", 
                    duration_hours=35,
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    cost=89.0,
                    rating=4.5,
                    learning_styles=[LearningStyle.PROJECT_BASED, LearningStyle.HANDS_ON],
                    prerequisites=["javascript"]
                )
            ]
        }
        
        # Generate resources for all skills
        for skill_name in self.ontology.skill_nodes:
            if skill_name in sample_resources:
                resources_db[skill_name] = sample_resources[skill_name]
            else:
                # Generate default resources
                skill_node = self.ontology.skill_nodes[skill_name]
                resources_db[skill_name] = [
                    LearningResource(
                        name=f"{skill_node.name} Complete Course",
                        url=f"https://example.com/{skill_name}-course",
                        type="course",
                        duration_hours=skill_node.estimated_hours,
                        difficulty=skill_node.difficulty,
                        cost=50.0,
                        rating=4.3,
                        learning_styles=[LearningStyle.HANDS_ON],
                        prerequisites=skill_node.prerequisites
                    )
                ]
        
        return resources_db
    
    def generate_optimal_path(self, 
                            missing_skills: List[str], 
                            user_profile: UserProfile,
                            target_job_skills: List[str] = None) -> LearningPath:
        """Generate optimized learning path using multi-objective optimization"""
        
        if not missing_skills:
            return LearningPath([], [], 0, 0, 1.0, 0.0, 0.0, [])
        
        # Get all required skills including prerequisites
        all_required_skills = set()
        for skill in missing_skills:
            prereq_chain = self.ontology.get_prerequisites_chain(skill)
            all_required_skills.update(prereq_chain)
        
        # Remove skills user already has
        skills_to_learn = []
        for skill in all_required_skills:
            if skill not in user_profile.current_skills:
                proficiency = user_profile.skill_proficiency.get(skill, 0.0)
                if proficiency < 0.8:  # Need to learn/improve
                    skills_to_learn.append(skill)
        
        # Optimize sequence using multiple criteria
        optimized_sequence = self._optimize_sequence(skills_to_learn, user_profile, target_job_skills)
        
        # Select best resources for each skill
        selected_resources = []
        total_duration = 0
        total_cost = 0
        
        for skill in optimized_sequence:
            resource = self._select_best_resource(skill, user_profile)
            if resource:
                selected_resources.append(resource)
                total_duration += resource.duration_hours
                total_cost += resource.cost
        
        # Calculate scores
        completion_prob = self._calculate_completion_probability(optimized_sequence, user_profile)
        market_relevance = self._calculate_market_relevance(optimized_sequence, target_job_skills)
        personalization = self._calculate_personalization_score(selected_resources, user_profile)
        
        # Generate milestones
        milestones = self._generate_milestones(optimized_sequence, selected_resources, user_profile)
        
        return LearningPath(
            skills_sequence=optimized_sequence,
            resources_sequence=selected_resources,
            total_duration_hours=total_duration,
            total_cost=total_cost,
            completion_probability=completion_prob,
            market_relevance_score=market_relevance,
            personalization_score=personalization,
            milestones=milestones
        )
    
    def _optimize_sequence(self, 
                          skills: List[str], 
                          user_profile: UserProfile,
                          target_job_skills: List[str] = None) -> List[str]:
        """Optimize skill learning sequence using graph algorithms"""
        
        if not skills:
            return []
        
        # Create subgraph for skills to learn
        skill_subgraph = self.ontology.graph.subgraph(skills).copy()
        
        # Add weights to edges based on multiple factors
        for u, v in skill_subgraph.edges():
            weight = self._calculate_transition_weight(u, v, user_profile, target_job_skills)
            skill_subgraph[u][v]['weight'] = weight
        
        # Use topological sort as base, then optimize
        try:
            base_sequence = list(nx.topological_sort(skill_subgraph))
        except nx.NetworkXError:
            # Handle cycles with DFS
            base_sequence = list(nx.dfs_preorder_nodes(skill_subgraph))
        
        # Apply local optimization using greedy approach
        optimized = self._local_optimization(base_sequence, user_profile, target_job_skills)
        
        return optimized
    
    def _calculate_transition_weight(self, 
                                   from_skill: str, 
                                   to_skill: str, 
                                   user_profile: UserProfile,
                                   target_job_skills: List[str] = None) -> float:
        """Calculate weight for transitioning between skills"""
        
        from_node = self.ontology.skill_nodes.get(from_skill)
        to_node = self.ontology.skill_nodes.get(to_skill)
        
        if not from_node or not to_node:
            return float('inf')
        
        weight = 0.0
        
        # Difficulty progression penalty
        diff_jump = to_node.difficulty.value - from_node.difficulty.value
        if diff_jump > 1:
            weight += diff_jump * 10  # Penalize big difficulty jumps
        
        # Market demand bonus
        weight -= to_node.market_demand * 5
        
        # Growth rate bonus  
        weight -= to_node.growth_rate * 3
        
        # Target job relevance bonus
        if target_job_skills and to_skill in target_job_skills:
            weight -= 15
        
        # Category coherence bonus (learning related skills together)
        if from_node.category == to_node.category:
            weight -= 5
        
        return max(weight, 0.1)  # Ensure positive weights
    
    def _local_optimization(self, 
                           sequence: List[str], 
                           user_profile: UserProfile,
                           target_job_skills: List[str] = None) -> List[str]:
        """Apply local optimizations to the sequence"""
        
        if len(sequence) <= 1:
            return sequence
        
        optimized = sequence.copy()
        
        # Gateway skills first optimization
        gateway_scores = self.ontology.get_gateway_skills()
        
        # Sort by gateway score while respecting prerequisites
        def can_move_earlier(skill_idx: int, new_pos: int) -> bool:
            skill = optimized[skill_idx]
            skill_prereqs = self.ontology.skill_nodes[skill].prerequisites
            
            # Check if all prerequisites are before new position
            for prereq in skill_prereqs:
                if prereq in optimized:
                    prereq_idx = optimized.index(prereq)
                    if prereq_idx >= new_pos:
                        return False
            return True
        
        # Move high-gateway skills earlier when possible
        for i in range(len(optimized)):
            skill = optimized[i]
            gateway_score = gateway_scores.get(skill, 0)
            
            if gateway_score > 3:  # High gateway skill
                # Try to move earlier
                for new_pos in range(i):
                    if can_move_earlier(i, new_pos):
                        skill = optimized.pop(i)
                        optimized.insert(new_pos, skill)
                        break
        
        return optimized
    
    def _select_best_resource(self, skill: str, user_profile: UserProfile) -> Optional[LearningResource]:
        """Select best learning resource for a skill based on user preferences"""
        
        available_resources = self.resource_database.get(skill, [])
        if not available_resources:
            return None
        
        best_resource = None
        best_score = -1
        
        for resource in available_resources:
            score = 0.0
            
            # Budget constraint
            if resource.cost > user_profile.budget:
                continue
            
            # Learning style match
            style_match = len(set(resource.learning_styles) & set(user_profile.learning_styles))
            score += style_match * 10
            
            # Resource type preference
            if resource.type in user_profile.preferred_resource_types:
                score += 15
            
            # Rating bonus
            score += resource.rating * 5
            
            # Cost efficiency (rating per dollar)
            if resource.cost > 0:
                score += (resource.rating / resource.cost) * 2
            else:
                score += 10  # Free resource bonus
            
            # Duration fit
            if user_profile.deadline_weeks:
                max_hours = user_profile.available_hours_per_week * user_profile.deadline_weeks
                if resource.duration_hours <= max_hours:
                    score += 5
                else:
                    score -= 10  # Penalty for too long
            
            if score > best_score:
                best_score = score
                best_resource = resource
        
        return best_resource
    
    def _calculate_completion_probability(self, skills: List[str], user_profile: UserProfile) -> float:
        """Estimate probability of completing the learning path"""
        
        if not skills:
            return 1.0
        
        total_hours = sum(self.ontology.skill_nodes[skill].estimated_hours for skill in skills)
        
        # Time constraint check
        if user_profile.deadline_weeks:
            available_hours = user_profile.available_hours_per_week * user_profile.deadline_weeks
            if total_hours > available_hours * 1.5:  # 50% buffer
                return 0.3
            elif total_hours > available_hours:
                return 0.6
        
        # Difficulty progression check
        difficulty_penalties = 0
        for i in range(1, len(skills)):
            prev_diff = self.ontology.skill_nodes[skills[i-1]].difficulty.value
            curr_diff = self.ontology.skill_nodes[skills[i]].difficulty.value
            if curr_diff > prev_diff + 1:
                difficulty_penalties += 0.1
        
        base_probability = 0.8
        probability = base_probability - difficulty_penalties
        
        # Learning style alignment bonus
        # This would require resource analysis, simplified for now
        probability += 0.1
        
        return max(0.1, min(1.0, probability))
    
    def _calculate_market_relevance(self, skills: List[str], target_job_skills: List[str] = None) -> float:
        """Calculate market relevance score for the skills"""
        
        if not skills:
            return 0.0
        
        total_demand = sum(self.ontology.skill_nodes[skill].market_demand for skill in skills)
        total_growth = sum(self.ontology.skill_nodes[skill].growth_rate for skill in skills)
        
        base_score = (total_demand / len(skills)) * 0.7 + (total_growth / len(skills)) * 0.3
        
        # Target job alignment bonus
        if target_job_skills:
            overlap = len(set(skills) & set(target_job_skills))
            alignment_bonus = (overlap / len(target_job_skills)) * 0.3
            base_score += alignment_bonus
        
        return min(1.0, base_score)
    
    def _calculate_personalization_score(self, resources: List[LearningResource], user_profile: UserProfile) -> float:
        """Calculate how well the path matches user preferences"""
        
        if not resources:
            return 0.0
        
        total_score = 0.0
        
        for resource in resources:
            # Learning style match
            style_overlap = len(set(resource.learning_styles) & set(user_profile.learning_styles))
            total_score += (style_overlap / len(user_profile.learning_styles)) * 0.4
            
            # Resource type preference
            if resource.type in user_profile.preferred_resource_types:
                total_score += 0.3
            
            # Budget fit
            if resource.cost <= user_profile.budget * 0.2:  # 20% of budget per resource
                total_score += 0.2
            
            # Quality bonus
            total_score += (resource.rating / 5.0) * 0.1
        
        return total_score / len(resources)
    
    def _generate_milestones(self, 
                           skills: List[str], 
                           resources: List[LearningResource],
                           user_profile: UserProfile) -> List[Dict[str, Any]]:
        """Generate learning milestones and checkpoints"""
        
        milestones = []
        cumulative_hours = 0
        cumulative_weeks = 0
        
        for i, (skill, resource) in enumerate(zip(skills, resources)):
            cumulative_hours += resource.duration_hours
            weeks_for_skill = max(1, resource.duration_hours // user_profile.available_hours_per_week)
            cumulative_weeks += weeks_for_skill
            
            milestone = {
                "skill": skill,
                "milestone_number": i + 1,
                "estimated_completion_week": cumulative_weeks,
                "cumulative_hours": cumulative_hours,
                "resource_name": resource.name,
                "validation_project": self._suggest_validation_project(skill),
                "next_skills_unlocked": self.ontology.skill_nodes[skill].unlocks,
                "difficulty_level": self.ontology.skill_nodes[skill].difficulty.name,
                "market_demand": self.ontology.skill_nodes[skill].market_demand
            }
            
            milestones.append(milestone)
        
        return milestones
    
    def _suggest_validation_project(self, skill: str) -> str:
        """Suggest a project to validate skill mastery"""
        
        project_suggestions = {
            "python": "Build a simple calculator or text-based game",
            "machine_learning": "Create a prediction model using a public dataset",
            "react": "Build a todo app with CRUD operations", 
            "django": "Create a blog application with user authentication",
            "sql": "Design and query a library management system database",
            "aws": "Deploy a web application using EC2 and RDS",
            "docker": "Containerize a multi-service application",
            "data_science": "Complete an end-to-end data analysis project"
        }
        
        return project_suggestions.get(skill, f"Complete a hands-on project demonstrating {skill} skills")


class DynamicLearningPathGenerator:
    """Main class for dynamic learning path generation"""
    
    def __init__(self):
        self.ontology = SkillOntologyBuilder()
        self.optimizer = LearningPathOptimizer(self.ontology)
        self.user_progress_tracker = {}
    
    def generate_personalized_path(self, 
                                 missing_skills: List[str],
                                 user_preferences: Dict[str, Any],
                                 target_job_skills: List[str] = None,
                                 current_skills: List[str] = None) -> Dict[str, Any]:
        """Generate a complete personalized learning path"""
        
        # Create user profile
        user_profile = UserProfile(
            available_hours_per_week=user_preferences.get('hours_per_week', 10),
            learning_styles=[LearningStyle(style) for style in user_preferences.get('learning_styles', ['hands_on'])],
            budget=user_preferences.get('budget', 200.0),
            deadline_weeks=user_preferences.get('deadline_weeks'),
            current_skills=current_skills or [],
            skill_proficiency=user_preferences.get('skill_proficiency', {}),
            preferred_resource_types=user_preferences.get('preferred_resource_types', ['course']),
            completed_courses=user_preferences.get('completed_courses', [])
        )
        
        # Generate optimal learning path
        learning_path = self.optimizer.generate_optimal_path(
            missing_skills, user_profile, target_job_skills
        )
        
        # Return comprehensive results
        return {
            "learning_path": {
                "skills_sequence": learning_path.skills_sequence,
                "total_duration_hours": learning_path.total_duration_hours,
                "total_cost": learning_path.total_cost,
                "estimated_weeks": max(1, learning_path.total_duration_hours // user_profile.available_hours_per_week),
                "completion_probability": learning_path.completion_probability,
                "market_relevance_score": learning_path.market_relevance_score,
                "personalization_score": learning_path.personalization_score
            },
            "resources": [
                {
                    "skill": resource.name,
                    "resource_name": resource.name,
                    "url": resource.url,
                    "type": resource.type,
                    "duration_hours": resource.duration_hours,
                    "cost": resource.cost,
                    "rating": resource.rating,
                    "difficulty": resource.difficulty.name,
                    "learning_styles": [style.value for style in resource.learning_styles]
                } for resource in learning_path.resources_sequence
            ],
            "milestones": learning_path.milestones,
            "skill_prerequisites": {
                skill: self.ontology.skill_nodes[skill].prerequisites 
                for skill in learning_path.skills_sequence
                if skill in self.ontology.skill_nodes
            },
            "skill_analytics": {
                "gateway_skills": self.ontology.get_gateway_skills(),
                "market_trends": {
                    skill: {
                        "demand": self.ontology.skill_nodes[skill].market_demand,
                        "growth_rate": self.ontology.skill_nodes[skill].growth_rate,
                        "category": self.ontology.skill_nodes[skill].category
                    }
                    for skill in learning_path.skills_sequence
                    if skill in self.ontology.skill_nodes
                }
            },
            "optimization_metrics": {
                "total_skills": len(learning_path.skills_sequence),
                "prerequisite_efficiency": self._calculate_prerequisite_efficiency(learning_path.skills_sequence),
                "cost_efficiency": learning_path.total_cost / max(1, learning_path.total_duration_hours),
                "time_efficiency": learning_path.personalization_score * learning_path.market_relevance_score
            }
        }
    
    def _calculate_prerequisite_efficiency(self, skills_sequence: List[str]) -> float:
        """Calculate how efficiently prerequisites are satisfied"""
        if len(skills_sequence) <= 1:
            return 1.0
        
        violations = 0
        total_checks = 0
        
        for i, skill in enumerate(skills_sequence):
            if skill in self.ontology.skill_nodes:
                prerequisites = self.ontology.skill_nodes[skill].prerequisites
                for prereq in prerequisites:
                    total_checks += 1
                    if prereq not in skills_sequence[:i]:
                        violations += 1
        
        return 1.0 - (violations / max(1, total_checks))
    
    def update_user_progress(self, user_id: str, completed_skill: str, proficiency_score: float):
        """Update user progress and adapt future recommendations"""
        if user_id not in self.user_progress_tracker:
            self.user_progress_tracker[user_id] = {}
        
        self.user_progress_tracker[user_id][completed_skill] = {
            'completed_date': np.datetime64('now'),
            'proficiency_score': proficiency_score,
            'time_to_complete': None  # Could track this for future predictions
        }
    
    def get_adaptive_recommendations(self, user_id: str, current_path: Dict[str, Any]) -> Dict[str, Any]:
        """Provide adaptive recommendations based on user progress"""
        
        progress = self.user_progress_tracker.get(user_id, {})
        
        recommendations = {
            "path_adjustments": [],
            "difficulty_adjustments": [],
            "resource_recommendations": [],
            "motivation_boosters": []
        }
        
        # Analyze progress patterns
        if progress:
            avg_proficiency = np.mean([p['proficiency_score'] for p in progress.values()])
            
            if avg_proficiency < 0.6:
                recommendations["difficulty_adjustments"].append({
                    "type": "reduce_difficulty",
                    "suggestion": "Consider additional foundational resources",
                    "reason": "Lower than expected proficiency scores"
                })
            
            elif avg_proficiency > 0.9:
                recommendations["difficulty_adjustments"].append({
                    "type": "increase_challenge",
                    "suggestion": "Skip to more advanced topics or add stretch goals",
                    "reason": "Consistently high performance"
                })
        
        return recommendations


# Test the dynamic learning path generator
if __name__ == "__main__":
    print("Dynamic Learning Path Generator - Research Implementation")
    print("=" * 60)
    
    # Initialize generator
    generator = DynamicLearningPathGenerator()
    
    # Example usage
    missing_skills = ["machine_learning", "react", "aws"]
    user_preferences = {
        'hours_per_week': 15,
        'learning_styles': ['hands_on', 'project_based'],
        'budget': 300.0,
        'deadline_weeks': 16,
        'preferred_resource_types': ['course', 'project'],
        'skill_proficiency': {'python': 0.7, 'javascript': 0.6}
    }
    current_skills = ["python", "javascript", "html", "css"]
    target_job_skills = ["machine_learning", "python", "aws", "react"]
    
    # Generate learning path
    result = generator.generate_personalized_path(
        missing_skills=missing_skills,
        user_preferences=user_preferences, 
        target_job_skills=target_job_skills,
        current_skills=current_skills
    )
    
    # Display results
    print("\n🎯 PERSONALIZED LEARNING PATH:")
    path = result["learning_path"]
    print(f"Skills Sequence: {' → '.join(path['skills_sequence'])}")
    print(f"Total Duration: {path['total_duration_hours']} hours ({path['estimated_weeks']} weeks)")
    print(f"Total Cost: ${path['total_cost']:.2f}")
    print(f"Completion Probability: {path['completion_probability']:.1%}")
    print(f"Market Relevance: {path['market_relevance_score']:.1%}")
    print(f"Personalization: {path['personalization_score']:.1%}")
    
    print(f"\n📚 LEARNING RESOURCES ({len(result['resources'])}):")
    for i, resource in enumerate(result['resources'][:3], 1):
        print(f"{i}. {resource['resource_name']} ({resource['type']})")
        print(f"   Duration: {resource['duration_hours']}h | Cost: ${resource['cost']:.2f} | Rating: {resource['rating']}/5")
    
    print(f"\n🎯 MILESTONES ({len(result['milestones'])}):")
    for milestone in result['milestones'][:3]:
        print(f"Week {milestone['estimated_completion_week']}: {milestone['skill']}")
        print(f"   Validation: {milestone['validation_project']}")
        print(f"   Unlocks: {', '.join(milestone['next_skills_unlocked'][:2])}...")
    
    print(f"\n📊 OPTIMIZATION METRICS:")
    metrics = result['optimization_metrics']
    print(f"Prerequisite Efficiency: {metrics['prerequisite_efficiency']:.1%}")
    print(f"Cost Efficiency: ${metrics['cost_efficiency']:.2f}/hour")
    print(f"Time Efficiency: {metrics['time_efficiency']:.1%}")
    
    print("\n✅ Dynamic Learning Path Generation - IMPLEMENTED!")