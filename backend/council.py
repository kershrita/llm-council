"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple, Optional
from .openrouter import query_models_parallel, query_model
from .config import (
    COUNCIL_MODELS,
    CHAIRMAN_MODEL,
    TITLE_MODEL,
    FALLBACK_MODELS,
    TITLE_REQUEST_TIMEOUT_SECONDS,
)


async def stage1_collect_responses(
    user_query: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question

    Returns:
        Tuple of (successful responses, failed model attempts)
    """
    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel
    responses = await query_models_parallel(
        COUNCIL_MODELS,
        messages,
        fallback_models=FALLBACK_MODELS,
    )

    # Format results
    stage1_results = []
    failed_models = []
    for requested_model, response in responses.items():
        if response.get("ok"):
            resolved_model = response.get("model", requested_model)
            stage1_results.append({
                "model": requested_model,
                "requested_model": requested_model,
                "actual_model": resolved_model,
                "response": response.get("content", ""),
            })
            continue

        failed_models.append({
            "model": requested_model,
            "status_code": response.get("status_code"),
            "error": response.get("error", "Unknown error"),
            "attempted_models": response.get("attempted_models", [requested_model]),
        })

    return stage1_results, failed_models


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of (rankings list, label_to_model mapping, failed model attempts)
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(
        COUNCIL_MODELS,
        messages,
        fallback_models=FALLBACK_MODELS,
    )

    # Format results
    stage2_results = []
    failed_models = []
    for requested_model, response in responses.items():
        if response.get("ok"):
            resolved_model = response.get("model", requested_model)
            full_text = response.get("content", "")
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": requested_model,
                "requested_model": requested_model,
                "actual_model": resolved_model,
                "ranking": full_text,
                "parsed_ranking": parsed,
            })
            continue

        failed_models.append({
            "model": requested_model,
            "status_code": response.get("status_code"),
            "error": response.get("error", "Unknown error"),
            "attempted_models": response.get("attempted_models", [requested_model]),
        })

    return stage2_results, label_to_model, failed_models


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2

    Returns:
        Tuple of (result payload, optional failure details)
    """
    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model
    response = await query_model(
        CHAIRMAN_MODEL,
        messages,
        fallback_models=FALLBACK_MODELS,
    )

    if not response.get("ok"):
        # Fallback if chairman fails
        failure = {
            "model": CHAIRMAN_MODEL,
            "status_code": response.get("status_code"),
            "error": response.get("error", "Unknown error"),
            "attempted_models": response.get("attempted_models", [CHAIRMAN_MODEL]),
        }
        return {
            "model": CHAIRMAN_MODEL,
            "requested_model": CHAIRMAN_MODEL,
            "actual_model": None,
            "response": "Error: Unable to generate final synthesis.",
        }, failure

    resolved_model = response.get("model", CHAIRMAN_MODEL)
    return {
        "model": CHAIRMAN_MODEL,
        "requested_model": CHAIRMAN_MODEL,
        "actual_model": resolved_model,
        "response": response.get("content", ""),
    }, None


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                labels = []
                for match in numbered_matches:
                    label_match = re.search(r'Response [A-Z]', match)
                    if label_match:
                        labels.append(label_match.group())
                return labels

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use a lightweight configured model for title generation.
    response = await query_model(
        TITLE_MODEL,
        messages,
        timeout=TITLE_REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
        fallback_models=FALLBACK_MODELS,
    )

    if not response.get("ok"):
        # Fallback to a generic title
        return "New Conversation"

    title = response.get("content", "New Conversation").strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    # Stage 1: Collect individual responses
    stage1_results, stage1_failures = await stage1_collect_responses(user_query)

    failures: Dict[str, List[Dict[str, Any]]] = {
        "stage1": stage1_failures,
        "stage2": [],
        "stage3": [],
    }

    # If no models responded successfully, return error
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {
            "requested_models": COUNCIL_MODELS,
            "failures": failures,
            "fallbacks": {
                "stage1": [],
                "stage2": [],
                "stage3": [],
            },
        }

    # Stage 2: Collect rankings
    stage2_results, label_to_model, stage2_failures = await stage2_collect_rankings(
        user_query,
        stage1_results,
    )
    failures["stage2"] = stage2_failures

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    stage3_result, stage3_failure = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results
    )
    if stage3_failure is not None:
        failures["stage3"] = [stage3_failure]

    stage1_fallbacks = [
        {
            "requested_model": result.get("requested_model") or result.get("model"),
            "used_model": result.get("actual_model") or result.get("model"),
        }
        for result in stage1_results
        if (result.get("requested_model") or result.get("model"))
        and (result.get("actual_model") or result.get("model"))
        and (result.get("requested_model") or result.get("model")) != (result.get("actual_model") or result.get("model"))
    ]
    stage2_fallbacks = [
        {
            "requested_model": result.get("requested_model") or result.get("model"),
            "used_model": result.get("actual_model") or result.get("model"),
        }
        for result in stage2_results
        if (result.get("requested_model") or result.get("model"))
        and (result.get("actual_model") or result.get("model"))
        and (result.get("requested_model") or result.get("model")) != (result.get("actual_model") or result.get("model"))
    ]
    stage3_fallbacks = []
    requested_stage3_model = stage3_result.get("requested_model") or stage3_result.get("model")
    actual_stage3_model = stage3_result.get("actual_model") or stage3_result.get("model")
    if requested_stage3_model and actual_stage3_model and requested_stage3_model != actual_stage3_model:
        stage3_fallbacks.append({
            "requested_model": requested_stage3_model,
            "used_model": actual_stage3_model,
        })

    # Prepare metadata
    metadata = {
        "requested_models": COUNCIL_MODELS,
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "failures": failures,
        "fallbacks": {
            "stage1": stage1_fallbacks,
            "stage2": stage2_fallbacks,
            "stage3": stage3_fallbacks,
        },
    }

    return stage1_results, stage2_results, stage3_result, metadata
