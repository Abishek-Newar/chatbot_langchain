const {initializeApp} = require('firebase/app');
const {getFirestore,collection,getDocs,doc, setDoc, updateDoc, getDoc} = require("firebase/firestore")
const {getDatabase,ref,get} = require("firebase/database")
const {GoogleGenerativeAI} = require("@google/generative-ai")
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const  {GoogleGenerativeAIEmbeddings} = require("@langchain/google-genai");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter")
const { loadQAStuffChain,RetrievalQAChain} = require("langchain/chains")
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai")
const fs = require("fs")
const path = require("path");
const { getStorage } = require('firebase/storage');
require("dotenv").config()

const firebaseConfig = {
    apiKey: process.env.apiKey,
    authDomain: process.env.authDomain,
    databaseURL: process.env.databaseURL,
    projectId: process.env.projectId,
    storageBucket: process.env.storageBucket,
    messagingSenderId: process.env.messagingSenderId,
    appId: process.env.appId,
    measurementId: process.env.measurementId
  };

const geminiApiKey = process.env.GOOGLE_API_KEY ;
console.log(geminiApiKey)
const genAI = new GoogleGenerativeAI(geminiApiKey);
const model = genAI.getGenerativeModel({model:"gemini-pro"});
const embeddings = new GoogleGenerativeAIEmbeddings({
    GoogleGenerativeAI: genAI
})

const app = initializeApp(firebaseConfig)
const db = getDatabase(app)
const firestore = getFirestore(app);
const storage= getStorage(app)


async function loadFaissIndexesFromFirebase(collectionName) {
    const collectionRef = collection(firestore, collectionName);
    const querySnapshot = await getDocs(collectionRef);
    const vectorStores = [];

    for (const docSnap of querySnapshot.docs) {
        const data = docSnap.data();
        console.log(data);
        const { indexURL, docstoreURL } = data;

        if (indexURL && docstoreURL) {
            const tempDir = `.faiss_temp_store/faiss_temp_${docSnap.id}`;
            if (!fs.existsSync(tempDir)) {
                fs.mkdirSync(tempDir, { recursive: true });
            }

            const indexPath = path.join(tempDir, 'faiss.index');
            const docstorePath = path.join(tempDir, 'docstore.json');

            const indexResponse = await fetch(indexURL);
            const indexBuffer = await indexResponse.arrayBuffer();
            fs.writeFileSync(indexPath, Buffer.from(indexBuffer));

            const docstoreResponse = await fetch(docstoreURL);
            const docstoreBuffer = await docstoreResponse.arrayBuffer();
            fs.writeFileSync(docstorePath, Buffer.from(docstoreBuffer));

            const vectorStore = await FaissStore.load(tempDir, embeddings);
            vectorStores.push(vectorStore);
        }
    }
    if (vectorStores.length === 0) {
        throw new Error("No valid FAISS indexes found in Firebase Storage.");
    }


    return {
        asRetriever: () => ({
            getRelevantDocuments: async (query) => {
                const allResults = await Promise.all(
                    vectorStores.map(vs => vs.asRetriever().getRelevantDocuments(query))
                );
                return allResults.flat();
            }
        })
    };
}

async function answerQuestion(vectorStore, question) {
    const model = new ChatGoogleGenerativeAI({
        apiKey: geminiApiKey,
        modelName: "gemini-2.0-flash",
        temperature: 0.7
    })

    const prompt = `
                You are a helpful assistant. Please follow these rules when answering inquiries:
                - If the message is an inquiry, answer it using only the provided information.
                - If unsure about the answer to an inquiry, state that your knowledge is limited to the specific information provided by this business.
                - If there are multiple inquiries in a message, answer them one by one.
                - Refuse to tell jokes.
                - Do not use names or sensitive data.
                -first analysize the vector data being provided to you

                Inquiry: ${question}
            `;

    const chain = RetrievalQAChain.fromLLM(
        model,
        vectorStore.asRetriever(),
    );
    const res = await chain.call({
        query: prompt
    })
    return res.text
}

async function runMain(question){
    try {
        

        const loadedVectorStore = await loadFaissIndexesFromFirebase('faiss_indexes')

        if(loadedVectorStore){
            const answer = await answerQuestion(loadedVectorStore,question);
            console.log("Answer:",answer)
        }else{
            console.log("could not load the faiss index from firebase")
        }
    } catch (error) {
        console.error("Error:",error)
    }
}
runMain("Hello, I'm trying to pay our balance, but the pop up window does not allow me to add my CC information. Multiple people have tried multiple browsers and cards. Can you help? Becky Kolb WashU School of Medicine Marketing and Communications 618-806-6069")