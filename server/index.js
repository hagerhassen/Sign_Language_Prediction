import express from 'express'
import cors from "cors"
import path from 'path'
import "dot-env"

const app=express()
app.use(cors())

app.use(express.static(path.resolve(process.cwd(),'..','client/build')))
app.use('/static',express.static("static"))


app.get('*',(req,res)=>{
    res.sendFile(path.resolve(process.cwd(),'..','client/build','index.html'))
})

app.listen(process.env.PORT,()=>{
    console.log(`server is running on port ${process.env.PORT}`)
})
