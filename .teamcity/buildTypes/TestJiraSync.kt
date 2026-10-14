package buildTypes

import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.dockerSupport
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.vcs

object TestJiraSync : BuildType({
    name = "Lint and test Jira sync scripts"
    description = "Runs lint, formatting, and unit tests for the Jira sync scripts."
    buildNumberPattern = "%build.vcs.number%"

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        script {
            name = "Ruff check"
            scriptContent = """
                pip install "./.teamcity/jira_sync[dev]"
                ruff check .teamcity/jira_sync
                ruff format --check .teamcity/jira_sync
            """.trimIndent()
            dockerImage = "containers.deltares.nl/docker-proxy/python:3.12-slim"
        }
        script {
            name = "Unit tests"
            scriptContent = """
                pip install "./.teamcity/jira_sync[dev]"
                pytest .teamcity/jira_sync/tests --junitxml=test-results.xml
            """.trimIndent()
            dockerImage = "containers.deltares.nl/docker-proxy/python:3.12-slim"
        }
    }

    triggers {
        vcs {
            triggerRules = """
                +:.teamcity/jira_sync/**
                +:.teamcity/**
            """.trimIndent()
            branchFilter = "+:*"
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "DOCKER_REGISTRY_HARBOR"
            }
        }
        xmlReport {
            reportType = XmlReport.XmlReportType.JUNIT
            rules = "test-results.xml"
        }
    }

    requirements {
        equals("teamcity.agent.jvm.os.name", "Linux")
        equals("docker.server.osType", "linux")
    }
})
