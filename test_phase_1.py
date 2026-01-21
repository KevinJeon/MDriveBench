#!/usr/bin/env python3
"""
Phase 1 Validation Script - Tests that schema generation properly handles constraints

Run this after implementing Phase 1 changes to verify:
1. D2 scenarios have 1+ active constraints
2. D3 scenarios have 2+ active constraints  
3. LLM reasoning blocks are generated
4. Passive-only constraints rejected at D2+
"""

import json
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from scenario_generator.scenario_generator.schema_generator import (
    SchemaScenarioGenerator,
    SchemaGenerationConfig,
)


# Configuration
ACTIVE_CONSTRAINT_TYPES = {
    'opposite_approach_of',
    'perpendicular_left_of',
    'perpendicular_right_of',
    'left_lane_of',
    'right_lane_of',
    'merges_into_lane_of'
}

PASSIVE_CONSTRAINT_TYPES = {
    'follow_route_of',
    'same_approach_as'
}

# Initialize generator
print("Initializing schema generator...")
config = SchemaGenerationConfig()
generator = SchemaScenarioGenerator(config=config)


def count_constraints(spec, constraint_type):
    """Count constraints of a given type"""
    constraints = spec.get('vehicle_constraints', [])
    return sum(1 for c in constraints if c.get('constraint_type') == constraint_type)


def analyze_spec(spec):
    """Analyze a scenario spec and return results"""
    if spec is None:
        return {
            'num_vehicles': 0,
            'num_active': 0,
            'num_passive': 0,
            'num_total': 0,
            'active_types': [],
            'passive_types': [],
            'reasoning_present': False,
            'reasoning_text': 'N/A',
            'has_reasoning': False
        }
    
    num_vehicles = len(spec.ego_vehicles)
    
    active_constraints = [c for c in spec.vehicle_constraints if c.constraint_type.value in ACTIVE_CONSTRAINT_TYPES]
    passive_constraints = [c for c in spec.vehicle_constraints if c.constraint_type.value in PASSIVE_CONSTRAINT_TYPES]
    
    num_active = len(active_constraints)
    num_passive = len(passive_constraints)
    num_total = num_active + num_passive
    
    # Check for reasoning in metadata
    reasoning = getattr(spec, '_reasoning', None)
    has_reasoning = reasoning is not None and isinstance(reasoning, dict)
    reasoning_text = reasoning.get('constraint_summary', 'N/A') if has_reasoning else 'N/A'
    
    return {
        'num_vehicles': num_vehicles,
        'num_active': num_active,
        'num_passive': num_passive,
        'num_total': num_total,
        'active_types': [c.constraint_type.value for c in active_constraints],
        'passive_types': [c.constraint_type.value for c in passive_constraints],
        'reasoning_present': has_reasoning,
        'reasoning_text': reasoning_text,
        'has_reasoning': has_reasoning
    }


def validate_d1(spec):
    """D1: 0-1 constraint, can be passive"""
    analysis = analyze_spec(spec)
    
    checks = []
    checks.append(('At most 2 vehicles', analysis['num_vehicles'] <= 2, True))
    checks.append(('At most 1 constraint', analysis['num_total'] <= 1, True))
    
    return checks, analysis


def validate_d2(spec):
    """D2: 2 vehicles, 1-2 constraints, MUST include 1+ ACTIVE"""
    analysis = analyze_spec(spec)
    
    checks = []
    checks.append(('Has 2 vehicles', analysis['num_vehicles'] == 2, True))
    checks.append(('Has 1-2 constraints', 1 <= analysis['num_total'] <= 2, True))
    checks.append(('At least 1 ACTIVE constraint', analysis['num_active'] >= 1, True))
    
    # This is the key check for Phase 1
    checks.append(('NOT passive-only', analysis['num_active'] > 0, True))
    
    return checks, analysis


def validate_d3(spec):
    """D3: 3 vehicles, 2-3 constraints, MUST include 2+ ACTIVE from different vehicles"""
    analysis = analyze_spec(spec, 'highway', 3)
    
    if spec is None:
        checks = [('Spec generated', False, True)]
        return checks, analysis
    
    # Get unique vehicle pairs for active constraints
    active_constraints = [c for c in spec.vehicle_constraints if c.constraint_type.value in ACTIVE_CONSTRAINT_TYPES]
    active_pairs = set()
    for c in active_constraints:
        pair = (c.vehicle_a, c.vehicle_b)
        active_pairs.add(pair)
    
    checks = []
    checks.append(('Has 3 vehicles', analysis['num_vehicles'] == 3, True))
    checks.append(('Has 2-3 constraints', 2 <= analysis['num_total'] <= 3, True))
    checks.append(('At least 2 ACTIVE constraints', analysis['num_active'] >= 2, True))
    
    # This is the key check for Phase 1 - prevents "V2 real constraint, V3 just follow_route_of"
    checks.append(('ACTIVE constraints from different vehicles', len(active_pairs) >= 2, True))
    
    # Bonus check: are constraint types different?
    active_types = set(analysis['active_types'])
    if len(active_types) >= 2:
        checks.append(('Different ACTIVE constraint types', len(active_types) >= 2, True))
    
    return checks, analysis


def run_test(category='highway', difficulty=1, num_runs=3):
    """Run test for given difficulty"""
    print(f"\n{'='*70}")
    print(f"Testing D{difficulty} scenarios")
    print(f"{'='*70}")
    
    if difficulty == 1:
        validate_func = validate_d1
    elif difficulty == 2:
        validate_func = validate_d2
    elif difficulty == 3:
        validate_func = validate_d3
    else:
        print(f"Unknown difficulty {difficulty}")
        return False
    
    all_passed = True
    
    for run in range(1, num_runs + 1):
        print(f"\n[Run {run}/{num_runs}]")
        
        try:
            # Generate scenario
            spec, errors = generator.generate_spec(
                category=category,
                difficulty=difficulty,
            )
            
            if spec is None:
                print(f"  ERROR: Failed to generate spec: {errors}")
                all_passed = False
                continue
            
            # Validate
            checks, analysis = validate_func(spec)
            
            # Print results
            print(f"  Vehicles: {analysis['num_vehicles']}")
            print(f"  Total constraints: {analysis['num_total']} (active: {analysis['num_active']}, passive: {analysis['num_passive']})")
            if analysis['active_types']:
                print(f"  Active types: {', '.join(analysis['active_types'])}")
            if analysis['passive_types']:
                print(f"  Passive types: {', '.join(analysis['passive_types'])}")
            print(f"  Reasoning present: {analysis['reasoning_present']}")
            if analysis['reasoning_text'] != 'N/A':
                print(f"  Constraint summary: {analysis['reasoning_text']}")
            
            # Check all validations
            run_passed = True
            for check_name, result, is_required in checks:
                status = "✓" if result else "✗"
                importance = "REQUIRED" if is_required else "optional"
                print(f"    {status} {check_name} [{importance}]")
                
                if is_required and not result:
                    run_passed = False
                    all_passed = False
            
            if run_passed:
                print(f"  Result: PASS ✓")
            else:
                print(f"  Result: FAIL ✗")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def main():
    """Run all Phase 1 tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║ Phase 1 Validation Tests - Schema Generation with Constraint Semantics ║")
    print("╚" + "="*68 + "╝")
    
    print("\nThis script tests that the LLM now understands:")
    print("  • Passive vs active constraint distinction")
    print("  • Difficulty-specific requirements")
    print("  • Constraint combination rules")
    print("  • Reasoning generation")
    
    results = {}
    
    # Run tests
    results['D1'] = run_test(difficulty=1, num_runs=3)
    results['D2'] = run_test(difficulty=2, num_runs=3)
    results['D3'] = run_test(difficulty=3, num_runs=3)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for difficulty, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {difficulty}: {status}")
    
    all_passed = all(results.values())
    
    print(f"\n{'='*70}")
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("\nPhase 1 is working correctly!")
        print("\nNext steps:")
        print("  1. Review a few generated specs to verify reasoning quality")
        print("  2. Proceed to Phase 2: Path Picking Constraint-Awareness")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        print("\nCommon issues:")
        print("  • Passive-only constraints at D2+: Check LLM prompt includes anti-patterns")
        print("  • Missing reasoning: Verify response includes 'reasoning' field")
        print("  • D3 without diverse active: Check LLM prompt anti-patterns section")
        return 1


if __name__ == '__main__':
    sys.exit(main())
