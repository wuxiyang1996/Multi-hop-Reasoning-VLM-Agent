# Stage modified docs and code (exclude __pycache__ and untracked trainer/ unless desired)
git add readme.md decision_agents/README.md decision_agents/agent.py decision_agents/agent_helper.py rag/embedding/qwen3_vl_embedding.py rag/embedding/text_embedder.py skill_agents/README.md skill_agents/__init__.py skill_agents/query.py skill_agents/tool_call_reward.py

git commit -m "docs: reorganize readme; merge intro into overview; add Experience synthesis section

- Overview: merge Introduction into Overview, add Contents links
- Sections: Environments, Data structure, Skill agent, Decision-making agent, Experience synthesis (new), Trainer code (6)
- Experience synthesis: separate top-level section with world_model implemented + ToDo
- Consolidated ToDo (unfinished) table at end
- RAG: embedder freeze in text_embedder/qwen3_vl; EpisodicMemoryStore/SkillQueryEngine use RAG"
