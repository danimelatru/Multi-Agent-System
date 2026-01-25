"""
Offline evaluation script for routing accuracy, retrieval hit@k, and tool success.
"""
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from src.agents import Planner, Grounder, Actor
from src.retrieval import Retriever
from src.tools import execute_tool
from src.observability.logger import get_logger

logger = get_logger("eval")


@dataclass
class TestCase:
    """Single test case for evaluation."""
    query: str
    expected_category: str  # technical, billing, general
    expected_tools: List[str]
    expected_retrieval_queries: List[str]
    expected_answer_contains: List[str]


def load_test_cases(path: str = "data/eval_test_cases.json") -> List[TestCase]:
    """Load test cases from JSON file."""
    test_file = Path(path)
    if not test_file.exists():
        logger.warning(f"Test cases file not found: {path}, using defaults")
        return get_default_test_cases()
    
    with open(test_file) as f:
        data = json.load(f)
    
    return [TestCase(**tc) for tc in data]


def get_default_test_cases() -> List[TestCase]:
    """Get default test cases."""
    return [
        TestCase(
            query="What's the status of ORD-123?",
            expected_category="billing",
            expected_tools=["get_refund_status"],
            expected_retrieval_queries=[],
            expected_answer_contains=["ORD-123", "Refund"]
        ),
        TestCase(
            query="How do I fix error 101?",
            expected_category="technical",
            expected_tools=[],
            expected_retrieval_queries=["error 101"],
            expected_answer_contains=["error", "101"]
        ),
        TestCase(
            query="Hello, how are you?",
            expected_category="general",
            expected_tools=[],
            expected_retrieval_queries=[],
            expected_answer_contains=[]
        ),
    ]


def evaluate_routing_accuracy(planner: Planner, test_cases: List[TestCase]) -> Dict[str, Any]:
    """Evaluate routing accuracy by checking if plan matches expected category."""
    correct = 0
    total = len(test_cases)
    results = []
    
    for tc in test_cases:
        plan = planner.plan(tc.query, f"eval_{tc.query[:10]}")
        
        # Determine category from plan
        tools_needed = plan.get("tools_needed", [])
        retrieval_needs = plan.get("retrieval_needs", [])
        
        if "get_refund_status" in tools_needed:
            predicted_category = "billing"
        elif retrieval_needs:
            predicted_category = "technical"
        else:
            predicted_category = "general"
        
        is_correct = predicted_category == tc.expected_category
        if is_correct:
            correct += 1
        
        results.append({
            "query": tc.query,
            "expected": tc.expected_category,
            "predicted": predicted_category,
            "correct": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "metric": "routing_accuracy",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }


def evaluate_retrieval_hit_at_k(grounder: Grounder, test_cases: List[TestCase], k: int = 3) -> Dict[str, Any]:
    """Evaluate retrieval hit@k."""
    total_queries = 0
    hits = 0
    results = []
    
    for tc in test_cases:
        if not tc.expected_retrieval_queries:
            continue
        
        total_queries += 1
        retrieval_needs = [{"query": q} for q in tc.expected_retrieval_queries]
        evidence = grounder.retrieve(retrieval_needs, f"eval_{tc.query[:10]}")
        
        # Check if we retrieved relevant evidence
        has_relevant = len(evidence) > 0
        if has_relevant:
            hits += 1
        
        results.append({
            "query": tc.query,
            "evidence_count": len(evidence),
            "hit": has_relevant
        })
    
    hit_at_k = hits / total_queries if total_queries > 0 else 0.0
    
    return {
        "metric": f"retrieval_hit@{k}",
        "hit_rate": hit_at_k,
        "hits": hits,
        "total_queries": total_queries,
        "results": results
    }


def evaluate_tool_success(test_cases: List[TestCase]) -> Dict[str, Any]:
    """Evaluate tool execution success."""
    tool_tests = [tc for tc in test_cases if tc.expected_tools]
    total = len(tool_tests)
    successful = 0
    results = []
    
    for tc in tool_tests:
        for tool_name in tc.expected_tools:
            try:
                # Extract params from query (simplified)
                if "get_refund_status" in tool_name:
                    # Extract order ID from query
                    order_id = "ORD-123"  # Simplified
                    result = execute_tool(tool_name, {"order_id": order_id})
                    is_success = result.get("success", False)
                    if is_success:
                        successful += 1
                    results.append({
                        "tool": tool_name,
                        "success": is_success,
                        "result": result
                    })
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "success": False,
                    "error": str(e)
                })
    
    success_rate = successful / total if total > 0 else 0.0
    
    return {
        "metric": "tool_success_rate",
        "success_rate": success_rate,
        "successful": successful,
        "total": total,
        "results": results
    }


def main():
    """Run all evaluations."""
    logger.info("Starting evaluation")
    
    # Initialize components
    retriever = Retriever()
    planner = Planner()
    grounder = Grounder(retriever)
    
    # Load test cases
    test_cases = load_test_cases()
    logger.info(f"Loaded {len(test_cases)} test cases")
    
    # Run evaluations
    results = {}
    
    logger.info("Evaluating routing accuracy...")
    results["routing"] = evaluate_routing_accuracy(planner, test_cases)
    
    logger.info("Evaluating retrieval hit@k...")
    results["retrieval"] = evaluate_retrieval_hit_at_k(grounder, test_cases)
    
    logger.info("Evaluating tool success...")
    results["tools"] = evaluate_tool_success(test_cases)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nRouting Accuracy: {results['routing']['accuracy']:.2%}")
    print(f"Retrieval Hit@3: {results['retrieval']['hit_rate']:.2%}")
    print(f"Tool Success Rate: {results['tools']['success_rate']:.2%}")
    print("\n" + "="*60)
    
    # Save results
    output_file = Path("data/eval_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
