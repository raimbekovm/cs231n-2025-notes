# Security Policy

This repository is a static Quarto site of CS231n lecture notes. It has no
backend, no user accounts, and does not process user-submitted data — the
main security surface is the build pipeline (Quarto, CI, and dependencies)
and the integrity of the published site.

## Supported Versions

Only the `main` branch and the site it publishes are maintained. There are no
versioned releases to patch separately.

## Reporting a Vulnerability

If you find a security issue — for example, a compromised dependency, a CI
workflow that could be abused, or a way to inject content into the published
site — please report it privately rather than opening a public issue, using
[GitHub's private vulnerability reporting](https://github.com/raimbekovm/cs231n-2025-notes/security/advisories/new).

Please include steps to reproduce and the potential impact. You can expect an
initial response within a few days.

Regular typos, broken links, or academic corrections are not security issues
— please file those as normal [issues](https://github.com/raimbekovm/cs231n-2025-notes/issues)
or see [CONTRIBUTING.md](CONTRIBUTING.md) instead.
