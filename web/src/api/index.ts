// The only module that imports the generated schema. The rest of the app
// imports types from `src/api`, never from `schema.d.ts` directly.
export type { components, operations, paths } from './schema'
export { api, apiRequest } from './client'
export { ApiError, rethrowHttp } from './errors'
