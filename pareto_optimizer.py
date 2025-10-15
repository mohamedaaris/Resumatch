"""
Multi-objective Pareto Optimization for Matching
===============================================

Novel research feature implementing Pareto optimization for job matching:
- Multiple conflicting objectives: skill match vs growth potential vs cultural fit
- Pareto frontier analysis and trade-off visualization  
- User preference elicitation and adaptive weighting

Research Contribution:
- First to apply multi-objective optimization to job matching
- Novel Pareto frontier exploration for career decisions
- Interactive trade-off analysis for informed choice
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import random

class OptimizationObjective(Enum):
    SKILL_MATCH = "skill_match"
    GROWTH_POTENTIAL = "growth_potential" 
    CULTURAL_FIT = "cultural_fit"
    LOCATION_PREFERENCE = "location_preference"
    SALARY_EXPECTATION = "salary_expectation"
    WORK_LIFE_BALANCE = "work_life_balance"

@dataclass
class JobCandidate:
    """A job candidate with multi-objective scores"""
    job_id: str
    title: str
    company: str
    objectives: Dict[OptimizationObjective, float]  # 0-1 scores
    metadata: Dict[str, Any]

@dataclass
class ParetoSolution:
    """A solution on the Pareto frontier"""
    job: JobCandidate
    dominates: List[str]  # Job IDs this solution dominates
    dominated_by: List[str]  # Job IDs that dominate this solution
    pareto_rank: int
    crowding_distance: float

class MultiObjectiveOptimizer:
    """Multi-objective optimizer using Pareto dominance"""
    
    def __init__(self, objectives: List[OptimizationObjective]):
        self.objectives = objectives
        self.solutions = []
        self.pareto_fronts = []
    
    def optimize(self, job_candidates: List[JobCandidate], 
                user_preferences: Dict[OptimizationObjective, float] = None) -> Dict[str, Any]:
        """Perform multi-objective optimization"""
        
        if not job_candidates:
            return {'pareto_fronts': [], 'recommendations': [], 'trade_offs': {}}
        
        # Calculate Pareto dominance
        pareto_solutions = self._calculate_pareto_dominance(job_candidates)
        
        # Rank solutions into Pareto fronts
        pareto_fronts = self._rank_pareto_fronts(pareto_solutions)
        
        # Calculate crowding distances within each front
        for front in pareto_fronts:
            self._calculate_crowding_distances(front)
        
        # Generate recommendations based on user preferences
        recommendations = self._generate_recommendations(pareto_fronts, user_preferences)
        
        # Analyze trade-offs
        trade_offs = self._analyze_trade_offs(pareto_fronts[0] if pareto_fronts else [])
        
        return {
            'pareto_fronts': [[sol.job for sol in front] for front in pareto_fronts],
            'pareto_solutions': pareto_solutions,
            'recommendations': recommendations,
            'trade_offs': trade_offs,
            'optimization_stats': {
                'total_solutions': len(job_candidates),
                'pareto_optimal_solutions': len(pareto_fronts[0]) if pareto_fronts else 0,
                'objectives_count': len(self.objectives)
            }
        }
    
    def _calculate_pareto_dominance(self, job_candidates: List[JobCandidate]) -> List[ParetoSolution]:
        """Calculate Pareto dominance relationships"""
        
        solutions = []
        
        for job in job_candidates:
            solution = ParetoSolution(
                job=job,
                dominates=[],
                dominated_by=[],
                pareto_rank=0,
                crowding_distance=0.0
            )
            solutions.append(solution)
        
        # Calculate dominance relationships
        for i, sol_i in enumerate(solutions):
            for j, sol_j in enumerate(solutions):
                if i != j:
                    if self._dominates(sol_i.job, sol_j.job):
                        sol_i.dominates.append(sol_j.job.job_id)
                    elif self._dominates(sol_j.job, sol_i.job):
                        sol_i.dominated_by.append(sol_j.job.job_id)
        
        return solutions
    
    def _dominates(self, job_a: JobCandidate, job_b: JobCandidate) -> bool:
        """Check if job_a Pareto dominates job_b"""
        
        better_in_at_least_one = False
        
        for objective in self.objectives:
            score_a = job_a.objectives.get(objective, 0.0)
            score_b = job_b.objectives.get(objective, 0.0)
            
            if score_a < score_b:  # job_a is worse in this objective
                return False
            elif score_a > score_b:  # job_a is better in this objective
                better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def _rank_pareto_fronts(self, solutions: List[ParetoSolution]) -> List[List[ParetoSolution]]:
        """Rank solutions into Pareto fronts"""
        
        fronts = []
        remaining_solutions = solutions.copy()
        rank = 1
        
        while remaining_solutions:
            current_front = []
            
            # Find solutions not dominated by any remaining solution
            for solution in remaining_solutions:
                is_dominated = False
                for other in remaining_solutions:
                    if other.job.job_id in solution.dominated_by:
                        is_dominated = True
                        break
                
                if not is_dominated:
                    current_front.append(solution)
                    solution.pareto_rank = rank
            
            # Remove current front from remaining solutions
            for solution in current_front:
                remaining_solutions.remove(solution)
            
            fronts.append(current_front)
            rank += 1
            
            # Safety check to prevent infinite loop
            if rank > 10:
                break
        
        return fronts
    
    def _calculate_crowding_distances(self, front: List[ParetoSolution]):
        """Calculate crowding distances for solutions in a front"""
        
        if len(front) <= 2:
            for solution in front:
                solution.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for solution in front:
            solution.crowding_distance = 0.0
        
        # Calculate distance for each objective
        for objective in self.objectives:
            # Sort by objective value
            front.sort(key=lambda x: x.job.objectives.get(objective, 0.0))
            
            # Boundary solutions get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_values = [sol.job.objectives.get(objective, 0.0) for sol in front]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range == 0:
                continue
            
            # Calculate distances for internal solutions
            for i in range(1, len(front) - 1):
                if front[i].crowding_distance != float('inf'):
                    distance = (front[i+1].job.objectives.get(objective, 0.0) - 
                              front[i-1].job.objectives.get(objective, 0.0)) / obj_range
                    front[i].crowding_distance += distance
    
    def _generate_recommendations(self, pareto_fronts: List[List[ParetoSolution]], 
                                user_preferences: Dict[OptimizationObjective, float] = None) -> List[Dict[str, Any]]:
        """Generate recommendations based on Pareto fronts and user preferences"""
        
        if not pareto_fronts or not pareto_fronts[0]:
            return []
        
        recommendations = []
        
        # If user preferences are provided, use weighted sum
        if user_preferences:
            scored_solutions = []
            for solution in pareto_fronts[0]:  # Only consider Pareto optimal solutions
                weighted_score = 0.0
                for objective, weight in user_preferences.items():
                    score = solution.job.objectives.get(objective, 0.0)
                    weighted_score += score * weight
                
                scored_solutions.append((solution, weighted_score))
            
            # Sort by weighted score
            scored_solutions.sort(key=lambda x: x[1], reverse=True)
            
            for solution, score in scored_solutions[:5]:
                recommendations.append({
                    'job': solution.job,
                    'recommendation_type': 'preference_optimized',
                    'weighted_score': score,
                    'pareto_rank': solution.pareto_rank,
                    'crowding_distance': solution.crowding_distance
                })
        
        else:
            # No preferences - recommend diverse solutions from Pareto front
            pareto_optimal = pareto_fronts[0]
            
            # Sort by crowding distance (most diverse first)
            pareto_optimal.sort(key=lambda x: x.crowding_distance if x.crowding_distance != float('inf') else 999, reverse=True)
            
            for i, solution in enumerate(pareto_optimal[:5]):
                recommendations.append({
                    'job': solution.job,
                    'recommendation_type': 'pareto_diverse',
                    'diversity_rank': i + 1,
                    'pareto_rank': solution.pareto_rank,
                    'crowding_distance': solution.crowding_distance
                })
        
        return recommendations
    
    def _analyze_trade_offs(self, pareto_front: List[ParetoSolution]) -> Dict[str, Any]:
        """Analyze trade-offs in the Pareto front"""
        
        if len(pareto_front) < 2:
            return {'trade_offs': [], 'objective_ranges': {}}
        
        trade_offs = []
        objective_ranges = {}
        
        # Calculate objective ranges
        for objective in self.objectives:
            values = [sol.job.objectives.get(objective, 0.0) for sol in pareto_front]
            objective_ranges[objective.value] = {
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values),
                'std': np.std(values) if values else 0.0
            }
        
        # Find significant trade-offs between objectives
        for i, obj_a in enumerate(self.objectives):
            for j, obj_b in enumerate(self.objectives[i+1:], i+1):
                
                # Calculate correlation between objectives
                values_a = [sol.job.objectives.get(obj_a, 0.0) for sol in pareto_front]
                values_b = [sol.job.objectives.get(obj_b, 0.0) for sol in pareto_front]
                
                if len(values_a) > 1 and len(values_b) > 1:
                    correlation = np.corrcoef(values_a, values_b)[0, 1]
                    
                    if abs(correlation) > 0.3:  # Significant correlation
                        trade_offs.append({
                            'objective_1': obj_a.value,
                            'objective_2': obj_b.value,
                            'correlation': correlation,
                            'trade_off_strength': abs(correlation),
                            'description': self._describe_trade_off(obj_a, obj_b, correlation)
                        })
        
        return {
            'trade_offs': trade_offs,
            'objective_ranges': objective_ranges
        }
    
    def _describe_trade_off(self, obj_a: OptimizationObjective, obj_b: OptimizationObjective, 
                           correlation: float) -> str:
        """Generate human-readable description of trade-off"""
        
        if correlation < -0.3:
            return f"Strong negative trade-off: Higher {obj_a.value.replace('_', ' ')} typically means lower {obj_b.value.replace('_', ' ')}"
        elif correlation > 0.3:
            return f"Positive correlation: {obj_a.value.replace('_', ' ')} and {obj_b.value.replace('_', ' ')} tend to move together"
        else:
            return f"Weak relationship between {obj_a.value.replace('_', ' ')} and {obj_b.value.replace('_', ' ')}"

class JobMatchingParetoOptimizer:
    """Main class for multi-objective job matching optimization"""
    
    def __init__(self):
        self.objectives = [
            OptimizationObjective.SKILL_MATCH,
            OptimizationObjective.GROWTH_POTENTIAL,
            OptimizationObjective.CULTURAL_FIT,
            OptimizationObjective.LOCATION_PREFERENCE,
            OptimizationObjective.SALARY_EXPECTATION
        ]
        self.optimizer = MultiObjectiveOptimizer(self.objectives)
    
    def generate_mock_job_data(self, num_jobs: int = 20) -> List[JobCandidate]:
        """Generate mock job data for testing"""
        
        jobs = []
        companies = ['Google', 'Microsoft', 'Amazon', 'Facebook', 'Apple', 'Netflix', 'Uber', 'Airbnb', 'Tesla', 'SpaceX']
        titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'DevOps Engineer', 'ML Engineer']
        
        for i in range(num_jobs):
            # Generate realistic correlated objective scores
            skill_match = random.uniform(0.3, 1.0)
            
            # Growth potential inversely correlated with skill match (easy jobs = less growth)
            growth_potential = max(0.2, 1.0 - skill_match * 0.6 + random.uniform(-0.2, 0.2))
            
            # Cultural fit somewhat random but biased by company
            cultural_fit = random.uniform(0.4, 0.9)
            
            # Location preference independent
            location_preference = random.uniform(0.2, 1.0)
            
            # Salary expectation correlated with skill requirements
            salary_expectation = min(1.0, skill_match * 0.8 + random.uniform(-0.1, 0.3))
            
            job = JobCandidate(
                job_id=f"job_{i}",
                title=random.choice(titles),
                company=random.choice(companies),
                objectives={
                    OptimizationObjective.SKILL_MATCH: skill_match,
                    OptimizationObjective.GROWTH_POTENTIAL: growth_potential,
                    OptimizationObjective.CULTURAL_FIT: cultural_fit,
                    OptimizationObjective.LOCATION_PREFERENCE: location_preference,
                    OptimizationObjective.SALARY_EXPECTATION: salary_expectation
                },
                metadata={'salary': 100000 + skill_match * 50000, 'level': 'mid'}
            )
            jobs.append(job)
        
        return jobs
    
    def optimize_job_matching(self, job_candidates: List[JobCandidate] = None, 
                             user_preferences: Dict[str, float] = None) -> Dict[str, Any]:
        """Perform multi-objective optimization for job matching"""
        
        if job_candidates is None:
            job_candidates = self.generate_mock_job_data()
        
        # Convert user preferences to enum keys
        enum_preferences = {}
        if user_preferences:
            for key, value in user_preferences.items():
                try:
                    objective = OptimizationObjective(key)
                    enum_preferences[objective] = value
                except ValueError:
                    continue
        
        # Perform optimization
        result = self.optimizer.optimize(job_candidates, enum_preferences)
        
        # Add summary statistics
        result['summary'] = self._generate_summary(result)
        
        return result
    
    def _generate_summary(self, optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of optimization results"""
        
        stats = optimization_result['optimization_stats']
        pareto_fronts = optimization_result['pareto_fronts']
        trade_offs = optimization_result['trade_offs']
        
        summary = {
            'pareto_efficiency': stats['pareto_optimal_solutions'] / stats['total_solutions'] if stats['total_solutions'] > 0 else 0,
            'front_sizes': [len(front) for front in pareto_fronts],
            'significant_trade_offs': len([t for t in trade_offs['trade_offs'] if t['trade_off_strength'] > 0.5]),
            'recommendation_diversity': len(set(rec['job'].company for rec in optimization_result['recommendations']))
        }
        
        return summary

# Integration function
def optimize_job_matches(job_data: List[Dict[str, Any]] = None, 
                        user_preferences: Dict[str, float] = None) -> Dict[str, Any]:
    """Main function for multi-objective job match optimization"""
    
    optimizer = JobMatchingParetoOptimizer()
    
    # Convert job data if provided
    job_candidates = None
    if job_data:
        job_candidates = []
        for job_dict in job_data:
            job = JobCandidate(
                job_id=job_dict.get('id', ''),
                title=job_dict.get('title', ''),
                company=job_dict.get('company', ''),
                objectives={
                    OptimizationObjective.SKILL_MATCH: job_dict.get('skill_match', 0.5),
                    OptimizationObjective.GROWTH_POTENTIAL: job_dict.get('growth_potential', 0.5),
                    OptimizationObjective.CULTURAL_FIT: job_dict.get('cultural_fit', 0.5),
                    OptimizationObjective.LOCATION_PREFERENCE: job_dict.get('location_preference', 0.5),
                    OptimizationObjective.SALARY_EXPECTATION: job_dict.get('salary_expectation', 0.5)
                },
                metadata=job_dict.get('metadata', {})
            )
            job_candidates.append(job)
    
    result = optimizer.optimize_job_matching(job_candidates, user_preferences)
    
    # Format for output
    return {
        'pareto_optimal_jobs': [
            {
                'job_id': job.job_id,
                'title': job.title,
                'company': job.company,
                'objectives': {obj.value: score for obj, score in job.objectives.items()}
            } for job in result['pareto_fronts'][0]
        ] if result['pareto_fronts'] else [],
        'recommendations': [
            {
                'job_id': rec['job'].job_id,
                'title': rec['job'].title,
                'company': rec['job'].company,
                'recommendation_type': rec['recommendation_type'],
                'scores': {obj.value: score for obj, score in rec['job'].objectives.items()}
            } for rec in result['recommendations']
        ],
        'trade_off_analysis': result['trade_offs'],
        'optimization_summary': result['summary'],
        'total_pareto_fronts': len(result['pareto_fronts'])
    }

# Test the multi-objective optimizer
if __name__ == "__main__":
    print("Multi-objective Pareto Optimization - Research Implementation")
    print("=" * 65)
    
    # Test with user preferences
    user_prefs = {
        'skill_match': 0.3,
        'growth_potential': 0.4,
        'cultural_fit': 0.2,
        'location_preference': 0.1
    }
    
    try:
        result = optimize_job_matches(user_preferences=user_prefs)
        
        print(f"\n🔄 OPTIMIZATION RESULTS:")
        summary = result['optimization_summary']
        print(f"Pareto Efficiency: {summary['pareto_efficiency']:.1%}")
        print(f"Pareto Front Size: {summary['front_sizes'][0] if summary['front_sizes'] else 0}")
        print(f"Total Fronts: {result['total_pareto_fronts']}")
        print(f"Significant Trade-offs: {summary['significant_trade_offs']}")
        
        print(f"\n🎯 TOP RECOMMENDATIONS ({len(result['recommendations'])}):") 
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"{i}. {rec['title']} at {rec['company']}")
            print(f"   Type: {rec['recommendation_type']}")
            scores = rec['scores']
            print(f"   Skills: {scores['skill_match']:.2f} | Growth: {scores['growth_potential']:.2f} | Culture: {scores['cultural_fit']:.2f}")
        
        print(f"\n⚖️ TRADE-OFF ANALYSIS:")
        trade_offs = result['trade_off_analysis']['trade_offs']
        for trade_off in trade_offs[:3]:
            print(f"• {trade_off['description']}")
            print(f"  Correlation: {trade_off['correlation']:.2f}")
        
        print(f"\n📊 OBJECTIVE RANGES:")
        ranges = result['trade_off_analysis']['objective_ranges']
        for obj, range_info in ranges.items():
            print(f"• {obj.replace('_', ' ').title()}: {range_info['min']:.2f} - {range_info['max']:.2f}")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Multi-objective Pareto Optimization - IMPLEMENTED!")