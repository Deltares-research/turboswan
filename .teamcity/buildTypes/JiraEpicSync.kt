package buildTypes

import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.schedule

object JiraEpicSync : BuildType({
    name = "Jira epic sync"
    maxRunningBuilds = 1

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        script {
            name = "Sync Jira and GitHub"
            scriptContent = """
                pip install ./.github/jira_sync
                python -m jira_sync.epic_sync
            """.trimIndent()
            dockerImage = "python:3.12-slim"
        }
    }

    triggers {
        schedule {
            schedulingPolicy = cron {
                minutes = "0/15"
            }
            branchFilter = "+:<default>"
            triggerBuild = always()
            withPendingChangesOnly = false
        }
    }
})
