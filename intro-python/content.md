# Introduction to Python

## Before You Begin

### Objectives

- 1
- 2

### Requirements

To complete this lab, you need the following account credentials and assets:

- Oracle Cloud Infrastructure Console login credentials and tenancy name
- Public IP Address and login credentials of the lab virtual machine (VM) for the hands-on exercises

## **STEP 0** : Create a New Jupyter Notebook

>**Note** : If like to use a prebuilt Jupyter notebook for this lab (so you don't have to copy/paste commands from this manual), proceed directly to the section **"How to run the lab steps using a pre-built Jupyter Notebook?"** in the **Appendix**.

1. In the lab VM, ensure that the notebook server is started and the dashboard is displayed as follows:

![](./images/notebook-dashboard.png " ")

2. Create a new Jupyter notebook. Click on the ![](./images/new-button.png) button at the top right and select the ![](./images/python3-button.png) kernel from the dropdown as shown below.

![](./images/new-notebook.png " ")

3. Or, if you are in another notebook, you need to click **File** -> **New Notebook** -> **Python3** as follows:

![](./images/file-new-notebook.png " ")

4. You will be presented a blank notebook.

![](./images/blank-notebook.png " ")

5. To use **Oracle Machine Learning for Python**, you must first import the ***oml*** Python module which contains the routines of OML4Py.

>**Note** : The ***oml*** module depends on few other Python modules, including ***cx_Oracle***, which is the module that enables Oracle Database access from Python.

6. Copy the below **"import oml"** Python statement by clicking the ![](./images/copy-button.png) button.

````
<copy>import oml</copy>
````

7. Paste the code in a blank cell of the notebook. Note the cell will be in **Edit mode** when the border is green.

![](./images/import-oml-before.png " ")

8. Run the statement by clicking the ![](./images/run-button.png) button. This will run the statement in the box where the cursor is currently active.

9. After the statement is run, a new blank cell created and the cell that was run get a number assigned in square brackets, and the green border moves to the next cell.

![](./images/import-oml-after.png " ")

10. Unless you plan on using the prebuilt Jupyter notebooks, please continue to use the above method of copying/pasting Python commands from the lab manual to the Jupyter notebook session in the lab VM.

## Summary

Using the Embedded Python Execution feature of OML4Py you can utilize the processing power of the Database server for running your ML tasks. You can also use third-party Python packages for and execute various third-party algorithms that are currently not supported within the Oracle Database. You have also seen the benefits of storing the Python scripts in the database.

## Appendix

### How to run this lab using a pre-built Jupyter Notebook?

Follow the below steps if you are short on time and choose to run the labs without copying/pasting the commands from this manual.

1. In the lab VM, ensure that the notebook server is started and the dashboard is displayed as follows:

![](./images/notebook-dashboard.png " ")

2. Or, if you are in another notebook, or cannot locate the notebook dashboard, click **File** -> **Open**.

![](./images/file-open.png " ")

3. Click on **saved_notebooks** and browse to the folder that contains the saved Python notebooks.

![](./images/click-saved-notebooks.png " ")

4. In the **saved-notebooks** folder click on **4-embedded-python-exec** lab.

![](./images/click-1-intro-python.png " ")

5. You will be presented the notebook for this lab. Run the entire notebook by clicking **Kernel** -> **Restart and Run All**. This will run all executable statements of the notebook (i.e. in the Python statements in the cells).

![](./images/restart-run-all.png " ")

6. Confirm that you like to **Restart and Run All Cells**. The notebook will take a few minutes to run.

![](./images/restart-and-run-all-confirm.png " ")

7. After all cells of the notebook successfully complete, you will see that each cell will get a number assigned in square brackets and (optionally) an output will be shown (also ensure there were no errors).

  Post completion, confirm that the last few cells of your notebook looks similar to the following:

![](./images/1-intro-python-complete.png " ")

8. You have successfully executed the notebook. You may now go through the notebook steps and inspect the individual commands and their respective outputs.
