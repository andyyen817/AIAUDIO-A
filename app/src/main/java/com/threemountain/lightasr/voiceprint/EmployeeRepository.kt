package com.threemountain.lightasr.voiceprint

import org.json.JSONArray
import org.json.JSONObject

class EmployeeRepository(
    private val storage: VoiceprintStorage,
) {
    @Synchronized
    fun createEmployee(name: String, storeName: String?): Employee {
        val cleanName = name.trim()
        require(cleanName.isNotEmpty()) { "员工姓名不能为空" }

        val employees = readEmployees().toMutableList()
        val now = System.currentTimeMillis()
        val employee = Employee(
            id = (employees.maxOfOrNull { it.id } ?: 0L) + 1L,
            name = cleanName,
            storeName = storeName?.trim()?.ifBlank { null },
            enrolled = false,
            sampleCount = 0,
            createdAt = now,
            updatedAt = now,
        )
        employees += employee
        writeEmployees(employees)
        return employee
    }

    @Synchronized
    fun listEmployees(): List<Employee> = readEmployees()

    @Synchronized
    fun getEmployee(employeeId: Long): Employee? {
        return readEmployees().firstOrNull { it.id == employeeId }
    }

    @Synchronized
    fun deleteEmployee(employeeId: Long) {
        writeEmployees(readEmployees().filterNot { it.id == employeeId })
    }

    @Synchronized
    fun updateSampleCount(employeeId: Long, sampleCount: Int) {
        updateEmployee(employeeId) { employee ->
            employee.copy(sampleCount = sampleCount, updatedAt = System.currentTimeMillis())
        }
    }

    @Synchronized
    fun markEnrolled(employeeId: Long, enrolled: Boolean) {
        updateEmployee(employeeId) { employee ->
            employee.copy(enrolled = enrolled, updatedAt = System.currentTimeMillis())
        }
    }

    @Synchronized
    fun clearEnrollment(employeeId: Long) {
        updateEmployee(employeeId) { employee ->
            employee.copy(enrolled = false, updatedAt = System.currentTimeMillis())
        }
    }

    @Synchronized
    private fun updateEmployee(employeeId: Long, transform: (Employee) -> Employee) {
        val employees = readEmployees().toMutableList()
        val index = employees.indexOfFirst { it.id == employeeId }
        require(index >= 0) { "员工不存在：$employeeId" }
        employees[index] = transform(employees[index])
        writeEmployees(employees)
    }

    private fun readEmployees(): List<Employee> {
        if (!storage.employeesFile.exists()) return emptyList()
        val content = storage.employeesFile.readText(Charsets.UTF_8).ifBlank { "[]" }
        val array = JSONArray(content)
        return List(array.length()) { index ->
            array.getJSONObject(index).toEmployee()
        }
    }

    private fun writeEmployees(employees: List<Employee>) {
        storage.ensureDirectories()
        val array = JSONArray()
        employees.forEach { array.put(it.toJson()) }
        storage.employeesFile.writeText(array.toString(2), Charsets.UTF_8)
    }

    private fun Employee.toJson(): JSONObject {
        return JSONObject()
            .put("id", id)
            .put("name", name)
            .put("storeName", storeName)
            .put("enrolled", enrolled)
            .put("sampleCount", sampleCount)
            .put("createdAt", createdAt)
            .put("updatedAt", updatedAt)
    }

    private fun JSONObject.toEmployee(): Employee {
        return Employee(
            id = getLong("id"),
            name = getString("name"),
            storeName = optString("storeName", "").ifBlank { null },
            enrolled = optBoolean("enrolled", false),
            sampleCount = optInt("sampleCount", 0),
            createdAt = optLong("createdAt", 0L),
            updatedAt = optLong("updatedAt", 0L),
        )
    }
}
