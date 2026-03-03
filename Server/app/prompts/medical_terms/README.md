# Medical Term Extraction Prompt Templates

## Overview

This directory contains prompt templates for extracting medical terminology from conversational patient responses using few-shot learning with Azure OpenAI GPT-4. The templates are designed to achieve >95% accuracy for domain-specific healthcare terminology extraction.

## Prompt Design Principles

### 1. Few-Shot Learning
- Each template includes 5-8 diverse examples covering common variations
- Examples demonstrate the expected input-output mapping
- Examples cover edge cases and ambiguous scenarios
- Examples include confidence scoring patterns

### 2. Structured Output
- All prompts use JSON structured output format
- Consistent schema across all medical term types
- Includes confidence scoring (0.0-1.0 range)
- Captures raw text snippet for audit trail

### 3. Context-Aware Extraction
- Field context explains the medical significance
- Valid values clearly enumerated
- Abbreviation mappings provided inline
- Validation rules explicitly stated

### 4. Variation Handling
- Abbreviations expanded before extraction
- Synonyms and colloquialisms mapped to standard terms
- Case-insensitive matching supported
- Multi-value scenarios handled (e.g., multiple locations)

## Prompt Templates

### hbsag_extraction.txt
**Purpose:** Extract HBsAg (Hepatitis B surface antigen) status

**Valid Values:**
- Positive
- Negative
- Unknown

**Variations Handled:**
- "Hep B positive" → Positive
- "HBV detected" → Positive
- "Hepatitis B negative" → Negative
- "No HBsAg" → Negative

**Accuracy Target:** >95%

### treatment_location_extraction.txt
**Purpose:** Extract treatment location information

**Valid Values:**
- OR (Operating Room)
- Bedside
- ICU-CCU (Intensive Care Unit - Cardiac Care Unit)
- ER (Emergency Room)
- Multi-Tx Room (Multiple Treatment Rooms)

**Variations Handled:**
- "Operating room" → OR
- "Surgery" → OR
- "Intensive care" → ICU-CCU
- "Emergency department" → ER
- "Bedside and OR" → Multi-Tx Room

**Accuracy Target:** >95%

### gender_extraction.txt
**Purpose:** Extract patient gender

**Valid Values:**
- Male
- Female

**Variations Handled:**
- "M" → Male
- "F" → Female
- "He was admitted" → Male
- "She presented" → Female
- "Gentleman" → Male
- "Woman" → Female

**Privacy Considerations:**
- Handle with sensitivity
- Extract only explicit or clearly implied information
- Respect patient confidentiality

**Accuracy Target:** >95%

## Configuration Files

### abbreviation_mappings.yaml
Bidirectional mappings for common medical abbreviations:
- HBsAg variations (Hep B, HBV, Hepatitis B)
- Treatment locations (OR, ICU, ER, etc.)
- Gender abbreviations (M, F)

**Usage:** Load before extraction to expand abbreviations

### field_schemas.yaml
Enum definitions and validation rules for each medical field:
- Valid enum values
- Fuzzy matching thresholds (0.85)
- Confidence thresholds (0.7)
- Validation rules

**Usage:** Validate LLM outputs against schemas

## Few-Shot Example Selection Criteria

### Quality Criteria
1. **Diversity:** Cover different phrasings and variations
2. **Clarity:** Unambiguous input-output mapping
3. **Relevance:** Realistic patient response patterns
4. **Edge Cases:** Include boundary scenarios (unknown, ambiguous)
5. **Confidence Calibration:** Show appropriate confidence levels

### Example Distribution
- 60% common/straightforward cases
- 30% variations and abbreviations
- 10% edge cases and ambiguous scenarios

### Example Structure
```
Input: <patient response>
Output: {"field": "value", "confidence": 0.XX, "raw_text": "snippet"}
```

## Prompt Testing Methodology

### 1. Unit Testing
- Test each prompt template with 20+ variations
- Measure extraction accuracy per template
- Validate structured output format
- Test confidence score calibration

### 2. Variation Testing
- Test all abbreviation mappings
- Test synonym handling
- Test case sensitivity
- Test multi-value scenarios

### 3. Edge Case Testing
- Ambiguous inputs
- Missing information
- Conflicting information
- Extremely long inputs (>1000 chars)

### 4. Accuracy Measurement
- Load 100+ test cases per field type
- Measure extraction accuracy: (correct / total) × 100
- Target: >95% accuracy
- Track false positives and false negatives

### 5. Consistency Testing
- Test same input multiple times
- Measure consistency across runs
- Target: >98% consistency

## Version Control Guidelines

### Versioning Scheme
- Major version: Breaking changes to output schema
- Minor version: New examples or improved accuracy
- Patch version: Bug fixes or clarifications

**Current Version:** 1.0.0

### Change Management
1. Test prompt changes with full test suite
2. Measure accuracy impact before deploying
3. Document changes in commit messages
4. Maintain backward compatibility when possible
5. Archive old versions for rollback capability

### Prompt Modification Process
1. Identify accuracy gap or edge case
2. Add new few-shot examples
3. Test with validation suite
4. Measure accuracy improvement
5. Deploy if >95% accuracy maintained
6. Update version number

## Accuracy Measurement Approach

### Metrics
1. **Extraction Accuracy:** % of correct extractions
2. **Confidence Calibration:** Correlation between confidence and correctness
3. **Consistency:** % of identical outputs for same input
4. **Variation Handling:** % of variations correctly mapped

### Measurement Process
1. Load test cases from `tests/data/medical_test_cases.yaml`
2. Run extraction for each test case
3. Compare extracted value to expected value
4. Calculate accuracy: (matches / total) × 100
5. Generate accuracy report with breakdown by field type

### Accuracy Targets
- Overall accuracy: >95%
- Per-field accuracy: >95%
- Variation handling: >90%
- Consistency: >98%

### Continuous Monitoring
- Track accuracy metrics in production
- Log validation failures for analysis
- Identify new variation patterns
- Update prompts quarterly or when accuracy drops below 95%

## Integration with Services

### SchemaValidationService
- Validates extracted values against `field_schemas.yaml`
- Detects hallucinated medical terms
- Triggers re-prompting for invalid values

### MedicalTermExtractionService
- Loads prompt templates from this directory
- Substitutes variables (raw_text, context)
- Calls Azure OpenAI with structured output
- Validates responses using SchemaValidationService

### Guardrails Integration
- Input sanitization before prompt injection
- Output validation against schemas
- Token budget enforcement
- Audit logging (no PII in logs)

## Troubleshooting Guide

### Low Accuracy (<95%)
1. Review failed test cases
2. Identify common failure patterns
3. Add few-shot examples for failure patterns
4. Test abbreviation mappings completeness
5. Adjust confidence thresholds

### Hallucinated Values
1. Strengthen validation rules in prompts
2. Add explicit "MUST be one of" constraints
3. Increase few-shot examples with valid values
4. Enable schema validation guardrails

### Inconsistent Extractions
1. Add more few-shot examples
2. Clarify ambiguous instructions
3. Strengthen output schema constraints
4. Test with multiple temperature settings

### Low Confidence Scores
1. Review confidence calibration in examples
2. Add more high-confidence examples
3. Clarify extraction criteria
4. Test with clearer input text

## Best Practices

1. **Keep prompts focused:** One field type per template
2. **Use clear language:** Avoid medical jargon in instructions
3. **Provide context:** Explain why the field matters
4. **Show examples:** 5-8 diverse few-shot examples minimum
5. **Validate outputs:** Always use structured JSON schemas
6. **Test thoroughly:** 100+ test cases per field type
7. **Monitor accuracy:** Track metrics in production
8. **Iterate regularly:** Update prompts based on failures
9. **Document changes:** Maintain version history
10. **Respect privacy:** Handle medical data with care

## References

- [Few-Shot Learning with LLMs](https://platform.openai.com/docs/guides/prompt-engineering/strategy-provide-examples)
- [Azure OpenAI Best Practices](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering)
- [HIPAA Medical Terminology Standards](https://www.hhs.gov/hipaa/for-professionals/index.html)
- [HL7 FHIR Terminology](https://www.hl7.org/fhir/terminologies.html)
- [Prompt Engineering Guide](https://www.promptingguide.ai/techniques/fewshot)

## Support

For questions or issues with prompt templates:
1. Review this README and test cases
2. Check accuracy metrics and logs
3. Consult the troubleshooting guide
4. Update prompts following version control guidelines
