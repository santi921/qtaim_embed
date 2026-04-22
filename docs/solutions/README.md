# Solutions Documentation

This directory contains documented solutions to problems encountered and solved in the QTAIM-Embed project. Each solution is structured for quick lookup when similar issues arise in the future.

## Categories

### Performance Issues

Solutions for training performance, optimization, and infrastructure bottlenecks.

#### [Optimizer Overhead Bottleneck: Fused Adam Implementation](performance-issues/optimizer-overhead-bottleneck-fused-adam.md)

**Date**: 2026-02-05 | **Severity**: High | **Status**: ✅ Resolved

**Problem**: Optimizer step consuming 31% of CPU time, creating major training bottleneck

**Solution**: Implemented GPU-accelerated fused Adam optimizer with graceful fallback

**Impact**: 20-40% faster training expected (31% → ~20% optimizer overhead)

**Quick Fix**:

```python
# Add to configure_optimizers():
try:
    optimizer = torch.optim.Adam(params, lr=lr, fused=True)
except RuntimeError:
    optimizer = torch.optim.Adam(params, lr=lr)
```

**Applicable When**:

- Optimizer CPU time > 25%
- PyTorch 2.0+
- CUDA available
- Using Adam/AdamW

---

## Quick Lookup

### By Symptom

**"Training is slow despite GPU availability"**
→ [Optimizer Overhead Bottleneck](performance-issues/optimizer-overhead-bottleneck-fused-adam.md)

**"Profiler shows high CPU time in optimizer"**
→ [Optimizer Overhead Bottleneck](performance-issues/optimizer-overhead-bottleneck-fused-adam.md)

**"Model is CPU-bound with low GPU utilization"**
→ [Optimizer Overhead Bottleneck](performance-issues/optimizer-overhead-bottleneck-fused-adam.md)

### By Component

**training-pipeline/optimizer**

- [Optimizer Overhead Bottleneck](performance-issues/optimizer-overhead-bottleneck-fused-adam.md)

### By Tag

**#performance**: All performance optimization solutions
**#gpu-acceleration**: GPU-related optimizations
**#pytorch**: PyTorch-specific solutions
**#profiling**: Issues identified through profiling

---

## Solution Template

When adding new solutions, use this structure:

```yaml
---
title: "Brief descriptive title"
date: YYYY-MM-DD
category: category-name
tags: [tag1, tag2, tag3]
severity: high/medium/low
component: module/submodule
status: resolved/in-progress/validated

symptoms:
  - Observable symptom 1
  - Observable symptom 2

root_cause: Brief explanation

solution_summary: One-sentence solution description

impact:
  before: Metrics before fix
  after: Metrics after fix
  improvement: Quantified improvement

affected_files:
  - file1.py
  - file2.py

prevention_strategies:
  - How to avoid this in future

applicable_contexts:
  - When this solution applies

not_applicable_when:
  - When NOT to use this solution
---

# Full detailed documentation...
```

---

## Usage Guidelines

### When to Document a Solution

Document solutions when:

- ✅ Problem is non-trivial (not a simple typo)
- ✅ Solution required investigation or research
- ✅ Problem could occur again in similar contexts
- ✅ Solution provides learning value for team

Don't document:

- ❌ Simple typos or obvious errors
- ❌ One-time issues with no reusability
- ❌ External tool bugs (file upstream issues instead)

### How to Find Solutions

1. **By symptom**: Search for error messages or observable behavior
2. **By component**: Browse category directories
3. **By tag**: Use frontmatter tags for filtering
4. **Full-text search**: `grep -r "search term" docs/solutions/`

### How to Add New Solutions

1. Create file in appropriate category directory
2. Use template structure above
3. Include YAML frontmatter for searchability
4. Cross-reference related documents
5. Add entry to this README

---

## 📈 Impact Summary

### Performance Optimizations Documented

| Solution        | Impact                 | Date       |
| --------------- | ---------------------- | ---------- |
| Fused Optimizer | +20-40% training speed | 2026-02-05 |

**Total cumulative speedup from documented solutions**: 20-40%

---

## Related Documentation

### Profiling & Performance

- [profiling/](../../profiling/) - Profiling results and analysis
- [docs/research/profiling-performance-patterns.md](../research/profiling-performance-patterns.md) - Comprehensive findings
- [docs/plans/](../plans/) - Optimization plans and strategies

### Research & Analysis

- [docs/research/](../research/) - Deep-dive research documents
- [docs/brainstorms/](../brainstorms/) - Design discussions and planning

---

## Contributing

When you solve a problem:

1. Document it immediately while context is fresh
2. Use the `/workflows:compound` command for assisted documentation
3. Include profiling data and before/after metrics
4. Cross-reference related issues and solutions
5. Update this README with new entry

**Remember**: Each documented solution compounds the team's knowledge. The first time costs research time; documented solutions cost only lookup time.

---

Last updated: 2026-02-05
