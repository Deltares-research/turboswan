import buildTypes.JiraEpicSync
import buildTypes.TestJiraSync
import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.dockerRegistry

version = "2026.2"

project {
    description = "Jira to github epic sync"

    params {
        param("env.JIRA_BASE_URL", "https://jira-knmi.atlassian.net")
        param("env.JIRA_EPIC_KEY", "YOUR-EPIC-KEY")
        param("env.GITHUB_REPOSITORY", "owner/repo")
        // Replace the credentialsJSON ids with tokens created under Project Settings > Tokens.
        password("env.JIRA_EMAIL", "credentialsJSON:REPLACE-WITH-JIRA-EMAIL-TOKEN", display = ParameterDisplay.HIDDEN)
        password("env.JIRA_API_TOKEN", "credentialsJSON:REPLACE-WITH-JIRA-API-TOKEN", display = ParameterDisplay.HIDDEN)
        password("env.GITHUB_TOKEN", "credentialsJSON:REPLACE-WITH-GITHUB-TOKEN", display = ParameterDisplay.HIDDEN)

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
            param("secure:client-secret", DslContext.getParameter("keeper-client-secret"))
            param("providerType", "teamcity-ksm")
        }
    }

    buildType(TestJiraSync)
    buildType(JiraEpicSync)
}
