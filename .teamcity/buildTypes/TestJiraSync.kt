package buildTypes

import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.vcs

object TestJiraSync : BuildType({
    name = "Lint and test Jira sync scripts"

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        script {
            name = "Ruff check"
            scriptContent = """
                pip install ruff
                ruff check .github/jira_sync
            """.trimIndent()
            dockerImage = "python:3.12-slim"
        }
        script {
            name = "Unit tests"
            scriptContent = """
                pip install "./.github/jira_sync[dev]"
                pytest .github/jira_sync/tests --junitxml=test-results.xml
            """.trimIndent()
            dockerImage = "python:3.12-slim"
        }
    }

    triggers {
        vcs {
            triggerRules = """
                +:.github/jira_sync/**
                +:.teamcity/**
            """.trimIndent()
            branchFilter = "+:*"
        }
    }

    features {
        xmlReport {
            reportType = XmlReport.XmlReportType.JUNIT
            rules = "test-results.xml"
        }
    }
})
