import buildTypes.JiraEpicSync
import buildTypes.TestJiraSync
import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.dockerRegistry

version = "2026.2"

project {
    description = "Jira to github epic sync"

    params {
        param("env.JIRA_BASE_URL", DslContext.getParameter("jira-base-url"))
        param("env.JIRA_EPIC_KEY", DslContext.getParameter("jira-epic-key"))
        param("env.GITHUB_REPOSITORY", DslContext.getParameter("github-repository"))
        password("jira-email", DslContext.getParameter("jira-email"), display = ParameterDisplay.HIDDEN)
        password("jira-api-token", DslContext.getParameter("jira-api-token"), display = ParameterDisplay.HIDDEN)
        password("env.JIRA_EMAIL", "%jira-email%", display = ParameterDisplay.HIDDEN)
        password("env.JIRA_API_TOKEN", "%jira-api-token%", display = ParameterDisplay.HIDDEN)
        password("env.GITHUB_TOKEN", DslContext.getParameter("github-token"), display = ParameterDisplay.HIDDEN)

        param("harbor-user", DslContext.getParameter("harbor-user"))
        password("harbor-secret", DslContext.getParameter("harbor-secret"))
    }

    features {
        dockerRegistry {
            id = "DOCKER_REGISTRY_HARBOR"
            name = "Harbor Docker registry"
            url = "https://containers.deltares.nl/"
            userName = "%harbor-user%"
            password = "%harbor-secret%"
        }
        feature {
            id = "PROJECT_EXT_1"
            type = "OAuthProvider"
            param("displayName", "Keeper Vault TurboSwan")
            param("secure:client-secret", "credentialsJSON:8a0243cc-5e81-41aa-ab85-43a6767d5cc5")
            param("providerType", "teamcity-ksm")
        }
    }

    buildType(TestJiraSync)
    buildType(JiraEpicSync)
}
