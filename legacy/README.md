# Legacy prototype (not used)

This folder holds the code from before the October 2026 rebuild: a FastAPI service, AWS CDK stacks,
Lambda handlers, an MCP server and a Streamlit app that showed mock data. The model in it was never
trained and the sample images were generated noise.

Nothing in the current app imports from here. It is kept so no history is lost on the rebuild
branch, and can be removed once that is approved. The same code is on `main` and on the tag
`backup/pre-rebuild-2026-10-23`.
